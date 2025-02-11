using System;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Linq;
using Bonsai;
using Bonsai.Dag;
using Bonsai.Expressions;
using Bonsai.Design;
using Bonsai.ML.Design;
using PointProcessDecoder.Core;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;
using static TorchSharp.torch;
using TorchSharp;

[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.ConditionalIntensitiesVisualizer), 
    Target = typeof(Bonsai.ML.PointProcessDecoder.Decode))]

namespace Bonsai.ML.PointProcessDecoder.Design
{
    public class ConditionalIntensitiesVisualizer : DialogTypeVisualizer
    {
        private int _rowCount = 1;
        public int RowCount
        {
            get => _rowCount;
            set
            {
                if (value < 1)
                {
                    throw new InvalidOperationException("The number of rows must be greater than 0.");
                }
                _rowCount = value;
                if (_container != null)
                {
                    _container.RowCount = _rowCount;
                }
            }
        }

        private int _columnCount = 1;
        public int ColumnCount
        {
            get => _columnCount;
            set
            {
                if (value < 1)
                {
                    throw new InvalidOperationException("The number of columns must be greater than 0.");
                }
                _columnCount = value;
                if (_container != null)
                {
                    _container.ColumnCount = _columnCount;
                }
            }
        }

        private int _pageCount = 1;
        public int PageCount => _pageCount;

        private int _selectedPageIndex = 0;
        public int SelectedPageIndex
        {
            get => _selectedPageIndex;
            set
            {
                if (value < 0 || value >= _pageCount)
                {
                    throw new InvalidOperationException("The selected page index is out of range.");
                }
                _selectedPageIndex = value;
            }
        }

        private PointProcessModel _model = null;
        private HeatMapSeriesOxyPlotBase[] _heatmapPlots = null;
        private long _conditionalIntensitiesCount = 0;
        private TableLayoutPanel _container = null;
        // create a dictionary to map the index of the heatmap plot to the corresponding conditional intensity
        private List<long> _conditionalIntensitiesCumulativeIndex = [];

        private StatusStrip _statusStrip = null;
        public StatusStrip StatusStrip => _statusStrip;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {

            Decode decodeNode = null;
            var expressionBuilderGraph = (ExpressionBuilderGraph)provider.GetService(typeof(ExpressionBuilderGraph));
            var typeVisualizerContext = (ITypeVisualizerContext)provider.GetService(typeof(ITypeVisualizerContext));
            if (expressionBuilderGraph != null && typeVisualizerContext != null)
            {
                decodeNode = ExpressionBuilder.GetWorkflowElement(
                    expressionBuilderGraph.Where(node => node.Value == typeVisualizerContext.Source)
                        .FirstOrDefault().Value) as Decode;
            }
  
            if (decodeNode == null)
            {
                Console.WriteLine("The decode node is invalid.");
                throw new InvalidOperationException("The decode node is invalid.");
            }

            var modelName = decodeNode.Model;
            if (string.IsNullOrEmpty(modelName))
            {
                Console.WriteLine("The point process model name is not set.");
                throw new InvalidOperationException("The point process model name is not set.");
            }

            _model = PointProcessModelManager.GetModel(modelName);

            if (_model == null)
            {
                Console.WriteLine($"The point process model with name {modelName} is not found.");
                throw new InvalidOperationException($"The point process model with name {modelName} is not found.");
            }

            if (_model.StateSpace.Dimensions != 2) 
            {
                Console.WriteLine("For the conditional intensities visualizer to work, the state space dimensions must be 2.");
                throw new InvalidOperationException("For the conditional intensities visualizer to work, the state space dimensions must be 2.");
            }
            
            _container = new TableLayoutPanel()
            {
                Dock = DockStyle.Fill,
                ColumnCount = _columnCount,
                RowCount = _rowCount,
            };

            InitializeHeatmaps();

            _pageCount = (int)Math.Ceiling((double)_conditionalIntensitiesCount / (_rowCount * _columnCount));

            var pageIndexLabel = new ToolStripLabel($"Page: {SelectedPageIndex}");
            var pageIndexControl = new ToolStripNumericUpDown()
            {
                Minimum = 0,
                Maximum = _pageCount - 1,
                DecimalPlaces = 0,
                Value = SelectedPageIndex,
            };

            // pageIndexControl.ValueChanged += (sender, e) =>
            // {
            //     var value = Convert.ToInt32(pageIndexControl.Value);
            //     try {
            //         SelectedPageIndex = value;
            //         UpdatePage();
            //         Show(null);
            //         pageIndexLabel.Text = $"Page: {SelectedPageIndex}";
            //     } catch (InvalidOperationException) {
            //         // pageIndexControl.Value = SelectedPageIndex;
            //     }
            // };

            var rowLabel = new ToolStripLabel($"Rows: {RowCount}");
            var rowControl = new ToolStripNumericUpDown()
            {
                Minimum = 1,
                DecimalPlaces = 0,
                Value = RowCount,
            };

            rowControl.ValueChanged += (sender, e) =>
            {
                var value = Convert.ToInt32(rowControl.Value);
                try
                {
                    RowCount = value;
                    _pageCount = (int)Math.Ceiling((double)_conditionalIntensitiesCount / (_rowCount * _columnCount));
                    UpdateTableLayout();
                    Show(null);
                    rowLabel.Text = $"Rows: {RowCount}";
                }
                catch (InvalidOperationException)
                {
                    // rowControl.Value = RowCount;
                }
            };

            var columnLabel = new ToolStripLabel($"Columns: {ColumnCount}");
            var columnControl = new ToolStripNumericUpDown()
            {
                Minimum = 1,
                DecimalPlaces = 0,
                Value = ColumnCount,
            };

            // columnControl.ValueChanged += (sender, e) =>
            // {
            //     var value = Convert.ToInt32(columnControl.Value);
            //     try
            //     {
            //         ColumnCount = value;
            //         _pageCount = (int)Math.Ceiling((double)_conditionalIntensitiesCount / (_rowCount * _columnCount));
            //         UpdatePage();
            //         Show(null);
            //         columnLabel.Text = $"Columns: {ColumnCount}";
            //     }
            //     catch (InvalidOperationException)
            //     {
            //         // columnControl.Value = ColumnCount;
            //     }
            // };

            _statusStrip = new StatusStrip()
            {
                Visible = true,
            };

            _statusStrip.Items.AddRange([
                pageIndexLabel, 
                pageIndexControl,
                rowLabel,
                rowControl,
                columnLabel,
                columnControl
            ]);

            UpdateHeatmaps();
            UpdateTableLayout();

            // _container.Controls.Add(statusStrip);
            // _container.MouseClick += (sender, e) => {
            //     if (e.Button == MouseButtons.Right) {
            //         statusStrip.Visible = !statusStrip.Visible;
            //     }
            // };

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            visualizerService?.AddControl(_container);
            visualizerService?.AddControl(_statusStrip);
        }

        private bool InitializeHeatmaps()
        {
            if (_model.Encoder.ConditionalIntensities.Length == 0 || (_model.Encoder.ConditionalIntensities.Length == 1 && _model.Encoder.ConditionalIntensities[0].numel() == 0))
            {
                return false;
            }

            _conditionalIntensitiesCount = 0;
            for (int i = 0; i < _model.Encoder.ConditionalIntensities.Length; i++) {
                var ci = _model.Encoder.ConditionalIntensities[i].clone();
                if (ci.IsInvalid) continue;
                if (ci.numel() > 0) {
                    _conditionalIntensitiesCount += ci.size(0);
                }
                _conditionalIntensitiesCumulativeIndex.Add(ci.shape[0] + _conditionalIntensitiesCumulativeIndex.LastOrDefault());
            }

            _heatmapPlots = new HeatMapSeriesOxyPlotBase[_conditionalIntensitiesCount];
            for (int i = 0; i < _conditionalIntensitiesCount; i++)
            {
                _heatmapPlots[i] = new HeatMapSeriesOxyPlotBase(0, 0)
                {
                    Dock = DockStyle.Fill,
                };
            }

            return true;
        }

        private void UpdateHeatmaps()
        {
            _container.Controls.Clear();
            // update heatmaps in container
            for (int i = 0; i < _rowCount; i++)
            {
                for (int j = 0; j < _columnCount; j++)
                {
                    var index = SelectedPageIndex * _rowCount * _columnCount + i * _columnCount + j;
                    if (index >= _conditionalIntensitiesCount)
                    {
                        break;
                    }

                    _container.Controls.Add(_heatmapPlots[index], j, i);
                }
            }
        }

        private void UpdateTableLayout()
        {
            var oldRowCount = _container.RowCount;
            var oldColumnCount = _container.ColumnCount;

            // update rows in container
            if (_rowCount > oldRowCount)
            {
                foreach (RowStyle rowStyle in _container.RowStyles)
                {
                    rowStyle.SizeType = SizeType.Percent;
                    rowStyle.Height = 100f / _rowCount;
                }
                
                _container.RowCount = _rowCount;
                for (int i = 0; i < _rowCount - oldRowCount; i++)
                {
                    _container.RowStyles.Add(new RowStyle(SizeType.Percent, 100f / _rowCount));
                    for (int j = 0; j < _columnCount; j++)
                    {
                        var index = SelectedPageIndex * _rowCount * _columnCount + (oldRowCount * oldColumnCount) + (i * _columnCount) + j;
                        if (index >= _conditionalIntensitiesCount)
                        {
                            break;
                        }

                        _container.Controls.Add(_heatmapPlots[index], j, i);
                    }
                }
            }
            else if (_rowCount < oldRowCount)
            {
                _container.RowCount = _rowCount;
                for (int i = _rowCount; i < oldRowCount; i++)
                {
                    _container.RowStyles.RemoveAt(_rowCount);
                    for (int j = 0; j < _columnCount; j++)
                    {
                        var index = SelectedPageIndex * _rowCount * _columnCount + (i * _columnCount) + j;
                        _container.Controls.Remove(_heatmapPlots[index]);
                    }
                }
            }

            // update columns in container
            if (_columnCount > oldColumnCount)
            {
                foreach (ColumnStyle columnStyle in _container.ColumnStyles)
                {
                    columnStyle.SizeType = SizeType.Percent;
                    columnStyle.Width = 100f / _columnCount;
                }
                _container.ColumnCount = _columnCount;
                for (int i = 0; i < _columnCount - oldColumnCount; i++)
                {
                    _container.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f / _columnCount));
                }

                // remove heatmaps in old rows
                for (int i = 1; i < _rowCount; i++)
                {
                    for (int j = 0; j < oldColumnCount; j++)
                    {
                        var index = SelectedPageIndex * _rowCount * _columnCount + (i * oldColumnCount) + j;
                        if (index >= _conditionalIntensitiesCount)
                        {
                            break;
                        }

                        _container.Controls.Remove(_heatmapPlots[index]);
                    }
                }

                // move heatmaps to the new columns
                for (int i = 0; i < _rowCount; i++)
                {
                    _container.RowStyles.Add(new RowStyle(SizeType.Percent, 100f / _rowCount));
                    for (int j = 0; j < _columnCount; j++)
                    {
                        if (i == 0 && j < oldColumnCount)
                        {
                            continue;
                        }

                        var index = SelectedPageIndex * _rowCount * _columnCount + (i * _columnCount) + j;
                        if (index >= _conditionalIntensitiesCount)
                        {
                            break;
                        }

                        _container.Controls.Add(_heatmapPlots[index], j, i);
                    }
                }
            }
            else if (_columnCount < oldColumnCount)
            {
                var oldColumnCount = _container.ColumnCount;
                _container.ColumnCount = _columnCount;
                for (int i = _columnCount; i < oldColumnCount; i++)
                {
                    _container.ColumnStyles.RemoveAt(_columnCount);
                }
            }
        }

        private (int ConditionalIntensitiesIndex, int ConditionalIntensitiesTensorIndex) GetConditionalIntensitiesIndex(int index)
        {

            var conditionalIntensitiesIndex = 0;
            for (int i = 0; i < _conditionalIntensitiesCumulativeIndex.Count; i++)
            {
                if (index < _conditionalIntensitiesCumulativeIndex[i])
                {
                    conditionalIntensitiesIndex = i;
                    break;
                }
            }
            var conditionalIntensitiesTensorIndex = conditionalIntensitiesIndex == 0 ? index : index - _conditionalIntensitiesCumulativeIndex[conditionalIntensitiesIndex - 1];
            return (conditionalIntensitiesIndex, (int)conditionalIntensitiesTensorIndex);
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            if (_heatmapPlots == null)
            {
                var success = InitializeHeatmaps();
                if (!success)
                {
                    throw new InvalidOperationException("The conditional intensities are empty.");
                }
            }
            
            var startIndex = SelectedPageIndex * _rowCount * _columnCount;
            var endIndex = Math.Min(startIndex + _rowCount * _columnCount, _conditionalIntensitiesCount);

            for (int i = startIndex; i < endIndex; i++) 
            {
                var (conditionalIntensitiesIndex, conditionalIntensitiesTensorIndex) = GetConditionalIntensitiesIndex(i);

                Array heatmapValues;

                var conditionalIntensity = _model.Encoder.ConditionalIntensities[conditionalIntensitiesIndex][conditionalIntensitiesTensorIndex]
                    .clone()
                    .to(CPU);

                if (conditionalIntensity.IsInvalid)
                {
                    continue;
                }

                if (conditionalIntensity.Dimensions == 3) {
                    conditionalIntensity = conditionalIntensity.sum(dim: 1);
                    Console.WriteLine($"ConditionalIntensity after sum: {conditionalIntensity}");
                }
                Console.WriteLine($"ConditionalIntensity: {conditionalIntensity}");
                var conditionalIntensityValues = conditionalIntensity
                    .to_type(ScalarType.Float64)
                    .data<double>();

                heatmapValues = conditionalIntensityValues
                    .ToNDArray();

                Console.WriteLine(heatmapValues.Length);
                var heatmap = new double[_model.StateSpace.Shape[0], _model.StateSpace.Shape[1]];
                Buffer.BlockCopy(heatmapValues, 0, heatmap, 0, heatmapValues.Length * sizeof(double));

                _heatmapPlots[i].UpdateHeatMapSeries(
                    0,
                    _model.StateSpace.Shape[0],
                    0,
                    _model.StateSpace.Shape[1],
                    heatmap
                );

                _heatmapPlots[i].UpdatePlot();
            }
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            if (_container != null)
            {
                if (!_container.IsDisposed)
                {
                    _container.Dispose();
                }
                _container = null;
            }

            for (int i = 0; i < _heatmapPlots.Length; i++)
            {
                if (!_heatmapPlots[i].IsDisposed)
                {
                    _heatmapPlots[i].Dispose();
                }
            }
            _heatmapPlots = null;

            if (_model != null)
            {
                _model.Dispose();
                _model = null;
            }

            _conditionalIntensitiesCount = 0;
            _rowCount = 1;
            _columnCount = 1;
            _pageCount = 1;
            _selectedPageIndex = 0;
            _conditionalIntensitiesCumulativeIndex.Clear();
        }
    }

    class ToolStripNumericUpDown : ToolStripControlHost
    {
        public ToolStripNumericUpDown()
            : base(new NumericUpDown())
        {
        }

        public NumericUpDown NumericUpDown
        {
            get { return Control as NumericUpDown; }
        }

        public int DecimalPlaces
        {
            get { return NumericUpDown.DecimalPlaces; }
            set { NumericUpDown.DecimalPlaces = value; }
        }

        public decimal Value
        {
            get { return NumericUpDown.Value; }
            set { NumericUpDown.Value = value; }
        }

        public decimal Minimum
        {
            get { return NumericUpDown.Minimum; }
            set { NumericUpDown.Minimum = value; }
        }

        public decimal Maximum
        {
            get { return NumericUpDown.Maximum; }
            set { NumericUpDown.Maximum = value; }
        }

        public event EventHandler ValueChanged
        {
            add { NumericUpDown.ValueChanged += value; }
            remove { NumericUpDown.ValueChanged -= value; }
        }
    }
}