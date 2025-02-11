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
using System.Reactive;

[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.ConditionalIntensitiesVisualizer), 
    Target = typeof(Bonsai.ML.PointProcessDecoder.Decode))]

namespace Bonsai.ML.PointProcessDecoder.Design
{
    public class ConditionalIntensitiesVisualizer : BufferedVisualizer
    {
        private int _rowCount = 1;
        public int RowCount
        {
            get => _rowCount;
            set
            {
                // if (value < 1)
                // {
                //     throw new InvalidOperationException("The number of rows must be greater than 0.");
                // }
                _rowCount = value;
                // if (_container != null)
                // {
                //     _container.RowCount = _rowCount;
                // }
            }
        }

        private int _columnCount = 1;
        public int ColumnCount
        {
            get => _columnCount;
            set
            {
                // if (value < 1)
                // {
                //     throw new InvalidOperationException("The number of columns must be greater than 0.");
                // }
                _columnCount = value;
                // if (_container != null)
                // {
                //     _container.ColumnCount = _columnCount;
                // }
            }
        }

        private int _selectedPageIndex = 0;
        public int SelectedPageIndex
        {
            get => _selectedPageIndex;
            set
            {
                // if (value < 0 || value >= _pageCount)
                // {
                //     throw new InvalidOperationException("The selected page index is out of range.");
                // }
                _selectedPageIndex = value;
            }
        }

        private int _pageCount = 1;
        private string _modelName = string.Empty;
        private PointProcessModel _model = null;
        private HeatMapSeriesOxyPlotBase[] _heatmapPlots = null;
        private long _conditionalIntensitiesCount = 0;
        private TableLayoutPanel _container = null;
        // create a dictionary to map the index of the heatmap plot to the corresponding conditional intensity
        private List<long> _conditionalIntensitiesCumulativeIndex = [];
        private StatusStrip _statusStrip = null;
        public StatusStrip StatusStrip => _statusStrip;
        private ToolStripNumericUpDown _pageIndexControl = null;
        private ToolStripNumericUpDown _rowControl = null;
        private ToolStripNumericUpDown _columnControl = null;

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

            _modelName = decodeNode.Model;
            if (string.IsNullOrEmpty(_modelName))
            {
                Console.WriteLine("The point process model name is not set.");
                throw new InvalidOperationException("The point process model name is not set.");
            }
            
            _container = new TableLayoutPanel()
            {
                Dock = DockStyle.Fill,
                ColumnCount = ColumnCount,
                RowCount = RowCount,
            };

            var pageIndexLabel = new ToolStripLabel($"Page: {SelectedPageIndex}");
            _pageIndexControl = new ToolStripNumericUpDown()
            {
                Minimum = 0,
                Maximum = _pageCount - 1,
                DecimalPlaces = 0,
                Value = SelectedPageIndex,
            };

            _pageIndexControl.ValueChanged += (sender, e) =>
            {
                var value = Convert.ToInt32(_pageIndexControl.Value);
                try {
                    SelectedPageIndex = value;
                    UpdateTableLayout();
                    Show(null);
                    pageIndexLabel.Text = $"Page: {SelectedPageIndex}";
                } catch (InvalidOperationException) { }
            };

            var rowLabel = new ToolStripLabel($"Rows: {RowCount}");
            _rowControl = new ToolStripNumericUpDown()
            {
                Minimum = 1,
                DecimalPlaces = 0,
                Value = RowCount,
            };

            _rowControl.ValueChanged += (sender, e) =>
            {
                var value = Convert.ToInt32(_rowControl.Value);
                try
                {
                    RowCount = value;
                    _pageCount = (int)Math.Ceiling((double)_conditionalIntensitiesCount / (_rowCount * _columnCount));
                    if (SelectedPageIndex >= _pageCount) 
                    {
                        SelectedPageIndex = _pageCount - 1;
                        _pageIndexControl.Maximum = _pageCount - 1;
                        _pageIndexControl.Value = SelectedPageIndex;
                    }
                    else 
                    {
                        UpdateTableLayout();
                        Show(null);
                    }
                    rowLabel.Text = $"Rows: {RowCount}";
                } catch (InvalidOperationException) { }
            };

            var columnLabel = new ToolStripLabel($"Columns: {ColumnCount}");
            _columnControl = new ToolStripNumericUpDown()
            {
                Minimum = 1,
                DecimalPlaces = 0,
                Value = ColumnCount,
            };

            _columnControl.ValueChanged += (sender, e) =>
            {
                var value = Convert.ToInt32(_columnControl.Value);
                try
                {
                    ColumnCount = value;
                    _pageCount = (int)Math.Ceiling((double)_conditionalIntensitiesCount / (_rowCount * _columnCount));
                    if (SelectedPageIndex >= _pageCount) 
                    {
                        SelectedPageIndex = _pageCount - 1;
                        _pageIndexControl.Maximum = SelectedPageIndex;
                        _pageIndexControl.Value = SelectedPageIndex;
                    }
                    else 
                    {
                        UpdateTableLayout();
                        Show(null);
                    }
                    columnLabel.Text = $"Columns: {ColumnCount}";
                } catch (InvalidOperationException) { }
            };

            _statusStrip = new StatusStrip()
            {
                Visible = true,
            };

            _statusStrip.Items.AddRange([
                pageIndexLabel, 
                _pageIndexControl,
                rowLabel,
                _rowControl,
                columnLabel,
                _columnControl
            ]);

            UpdateTableLayout();

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            visualizerService?.AddControl(_container);
            visualizerService?.AddControl(_statusStrip);
        }

        private bool InitializeModel()
        {
            try {
                _model = PointProcessModelManager.GetModel(_modelName);
            } catch {
                return false;
            }

            if (_model == null)
            {
                return false;
            }

            if (_model.StateSpace.Dimensions != 2) 
            {
                throw new InvalidOperationException("For the conditional intensities visualizer to work, the state space dimensions must be 2.");
            }

            if (_model.Encoder.ConditionalIntensities.Length == 0 || (_model.Encoder.ConditionalIntensities.Length == 1 && _model.Encoder.ConditionalIntensities[0].numel() == 0))
            {
                _model = null;
                return false;
            }

            InitializeHeatmaps();

            return true;
        }

        private bool InitializeHeatmaps()
        {
            _conditionalIntensitiesCount = 0;
            try {
                for (int i = 0; i < _model.Encoder.ConditionalIntensities.Length; i++) {
                    var ci = _model.Encoder.ConditionalIntensities[i];
                    if (_model.Encoder.ConditionalIntensities[i].numel() > 0) {
                        var size = _model.Encoder.ConditionalIntensities[i].size(0);
                        _conditionalIntensitiesCount += size;
                        _conditionalIntensitiesCumulativeIndex.Add(size + _conditionalIntensitiesCumulativeIndex.LastOrDefault());
                    }
                }
            } catch {
                return false;
            }

            _heatmapPlots = new HeatMapSeriesOxyPlotBase[_conditionalIntensitiesCount];
            for (int i = 0; i < _conditionalIntensitiesCount; i++)
            {
                _heatmapPlots[i] = new HeatMapSeriesOxyPlotBase(0, 0)
                {
                    Dock = DockStyle.Fill,
                };
            }

            _pageCount = (int)Math.Ceiling((double)_conditionalIntensitiesCount / (_rowCount * _columnCount));
            _pageIndexControl.Maximum = _pageCount - 1;

            return true;
        }

        private void UpdateTableLayout()
        {
            _container.Controls.Clear();
            _container.RowStyles.Clear();
            _container.ColumnStyles.Clear();

            _container.RowCount = RowCount;
            _container.ColumnCount = ColumnCount;

            for (int i = 0; i < RowCount; i++)
            {
                _container.RowStyles.Add(new RowStyle(SizeType.Percent, 100f / RowCount));
            }

            for (int i = 0; i < ColumnCount; i++)
            {
                _container.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f / ColumnCount));
            }

            for (int i = 0; i < RowCount; i++)
            {
                for (int j = 0; j < ColumnCount; j++)
                {
                    var index = SelectedPageIndex * RowCount * ColumnCount + i * ColumnCount + j;
                    if (index >= _conditionalIntensitiesCount)
                    {
                        break;
                    }

                    _container.Controls.Add(_heatmapPlots[index], j, i);
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

        protected override void ShowBuffer(IList<Timestamped<object>> values)
        {
            try { Show(values.LastOrDefault().Value); } 
            catch { }
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            if (_model is null)
            {
                var success = InitializeModel();
                if (!success)
                {
                    return;
                }
                UpdateTableLayout();
            }
            
            var startIndex = SelectedPageIndex * _rowCount * _columnCount;
            var endIndex = Math.Min(startIndex + _rowCount * _columnCount, _conditionalIntensitiesCount);

            for (int i = startIndex; i < endIndex; i++) 
            {
                var (conditionalIntensitiesIndex, conditionalIntensitiesTensorIndex) = GetConditionalIntensitiesIndex(i);

                Array heatmapValues;

                try {
                    var conditionalIntensity = _model.Encoder.ConditionalIntensities[conditionalIntensitiesIndex][conditionalIntensitiesTensorIndex];

                    if (conditionalIntensity.Dimensions == 2) {
                        conditionalIntensity = conditionalIntensity
                            .sum(dim: 0)
                            .exp();
                    }

                    var conditionalIntensityValues = conditionalIntensity
                        .to_type(ScalarType.Float64)
                        .data<double>();

                    heatmapValues = conditionalIntensityValues
                        .ToNDArray();
                } catch {
                    throw new InvalidOperationException("Error while updating the heatmap.");
                }

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
            _model = null;
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