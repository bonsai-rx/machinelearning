using System;
using System.Reactive.Linq;
using System.Reactive;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Linq;

using Bonsai;
using Bonsai.Expressions;
using Bonsai.Design;
using Bonsai.ML.Design;

using static TorchSharp.torch;

using PointProcessDecoder.Core;

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
            }
        }

        private int _selectedPageIndex = 0;
        public int SelectedPageIndex
        {
            get => _selectedPageIndex;
            set
            {
                _selectedPageIndex = value;
            }
        }

        private readonly int _sampleFrequency = 30;
        private int _pageCount = 1;
        private string _modelName = string.Empty;
        private List<HeatMapSeriesOxyPlotBase> _heatmapPlots = null;
        private int _conditionalIntensitiesCount = 0;
        private TableLayoutPanel _container = null;
        private readonly List<long> _conditionalIntensitiesCumulativeIndex = [];
        private StatusStrip _statusStrip = null;
        public StatusStrip StatusStrip => _statusStrip;
        private ToolStripNumericUpDown _pageIndexControl = null;
        private ToolStripNumericUpDown _rowControl = null;
        private ToolStripNumericUpDown _columnControl = null;
        private Tensor[] _conditionalIntensities = null;
        private long _stateSpaceWidth;
        private long _stateSpaceHeight;

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
                RowCount = _rowCount,
            };

            var pageIndexLabel = new ToolStripLabel($"Page: {_selectedPageIndex}");
            _pageIndexControl = new ToolStripNumericUpDown()
            {
                Minimum = 0,
                DecimalPlaces = 0,
                Value = _selectedPageIndex,
            };

            _pageIndexControl.ValueChanged += (sender, e) =>
            {
                var value = Convert.ToInt32(_pageIndexControl.Value);
                SelectedPageIndex = value;
                UpdateTableLayout();
                Show(null);
                pageIndexLabel.Text = $"Page: {_selectedPageIndex}";
            };

            var rowLabel = new ToolStripLabel($"Rows: {_rowCount}");
            _rowControl = new ToolStripNumericUpDown()
            {
                Minimum = 1,
                DecimalPlaces = 0,
                Value = _rowCount,
            };

            _rowControl.ValueChanged += (sender, e) =>
            {
                RowCount = Convert.ToInt32(_rowControl.Value);
                UpdatePages();
                if (_selectedPageIndex >= _pageCount) 
                {
                    SelectedPageIndex = _pageCount - 1;
                    _pageIndexControl.Value = _selectedPageIndex;
                }
                else 
                {
                    UpdateTableLayout();
                    Show(null);
                }
                rowLabel.Text = $"Rows: {_rowCount}";
            };

            var columnLabel = new ToolStripLabel($"Columns: {_columnCount}");
            _columnControl = new ToolStripNumericUpDown()
            {
                Minimum = 1,
                DecimalPlaces = 0,
                Value = _columnCount,
            };

            _columnControl.ValueChanged += (sender, e) =>
            {
                ColumnCount = Convert.ToInt32(_columnControl.Value);
                UpdatePages();
                if (_selectedPageIndex >= _pageCount) 
                {
                    SelectedPageIndex = _pageCount - 1;
                    _pageIndexControl.Value = _selectedPageIndex;
                }
                else 
                {
                    UpdateTableLayout();
                    Show(null);
                }
                columnLabel.Text = $"Columns: {_columnCount}";
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

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            visualizerService?.AddControl(_container);
            visualizerService?.AddControl(_statusStrip);
        }

        private void UpdatePages()
        {
            _pageCount = (int)Math.Ceiling((double)_conditionalIntensitiesCount / (_rowCount * _columnCount));
            _pageIndexControl.Maximum = _pageCount - 1;
        }

        private bool UpdateModel()
        {
            PointProcessModel model;

            try {
                model = PointProcessModelManager.GetModel(_modelName);
            } catch {
                return false;
            }

            if (model == null)
            {
                return false;
            }

            if (model.StateSpace.Dimensions != 2) 
            {
                throw new InvalidOperationException("For the conditional intensities visualizer to work, the state space dimensions must be 2.");
            }

            if (model.Encoder.ConditionalIntensities.Length == 0 || (model.Encoder.ConditionalIntensities.Length == 1 && model.Encoder.ConditionalIntensities[0].numel() == 0))
            {
                return false;
            }

            _conditionalIntensities = model.Encoder.ConditionalIntensities;
            _stateSpaceWidth = model.StateSpace.Shape[0];
            _stateSpaceHeight = model.StateSpace.Shape[1];

            return true;
        }

        private static int GetConditionalIntensitiesCount(Tensor[] conditionalIntensities, List<long> conditionalIntensitiesCumulativeIndex)
        {
            long conditionalIntensitiesCount = 0;
            conditionalIntensitiesCumulativeIndex.Clear();
            for (int i = 0; i < conditionalIntensities.Length; i++) {
                if (conditionalIntensities[i].numel() > 0) {
                    conditionalIntensitiesCount += conditionalIntensities[i].size(0);
                    conditionalIntensitiesCumulativeIndex.Add(conditionalIntensitiesCount);
                }
            }
            return (int)conditionalIntensitiesCount;
        }

        private bool UpdateHeatmaps()
        {
            if (_heatmapPlots is null)
            {
                _heatmapPlots = [];
                for (int i = 0; i < _conditionalIntensitiesCount; i++)
                {
                    _heatmapPlots.Add(new HeatMapSeriesOxyPlotBase(0, 0)
                    {
                        Dock = DockStyle.Fill,
                    });
                }
            }
            else if (_heatmapPlots.Count > _conditionalIntensitiesCount)
            {
                var count = _heatmapPlots.Count - _conditionalIntensitiesCount;
                for (int i = 0; i < count; i++)
                {
                    if (!_heatmapPlots[i + _conditionalIntensitiesCount].IsDisposed)
                    {
                        _heatmapPlots[i + _conditionalIntensitiesCount].Dispose();
                    }
                }
                _heatmapPlots.RemoveRange(_conditionalIntensitiesCount, count);
            }
            else if (_heatmapPlots.Count < _conditionalIntensitiesCount)
            {
                for (int i = _heatmapPlots.Count; i < _conditionalIntensitiesCount; i++)
                {
                    _heatmapPlots.Add(new HeatMapSeriesOxyPlotBase(0, 0)
                    {
                        Dock = DockStyle.Fill,
                    });
                }
            }

            return true;
        }

        private void UpdateTableLayout()
        {
            _container.Controls.Clear();
            _container.RowStyles.Clear();
            _container.ColumnStyles.Clear();

            _container.RowCount = _rowCount;
            _container.ColumnCount = _columnCount;

            for (int i = 0; i < _rowCount; i++)
            {
                _container.RowStyles.Add(new RowStyle(SizeType.Percent, 100f / _rowCount));
            }

            for (int i = 0; i < _columnCount; i++)
            {
                _container.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f / _columnCount));
            }

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
            var startIndex = SelectedPageIndex * _rowCount * _columnCount;
            var endIndex = Math.Min(startIndex + _rowCount * _columnCount, _conditionalIntensitiesCount);

            for (int i = startIndex; i < endIndex; i++) 
            {
                var (conditionalIntensitiesIndex, conditionalIntensitiesTensorIndex) = GetConditionalIntensitiesIndex(i);

                var conditionalIntensity = _conditionalIntensities[conditionalIntensitiesIndex][conditionalIntensitiesTensorIndex];

                if (conditionalIntensity.Dimensions == 2) {
                    conditionalIntensity = conditionalIntensity
                        .sum(dim: 0);
                }



                var conditionalIntensityValues = (double[,])conditionalIntensity
                    .exp()
                    .to_type(ScalarType.Float64)
                    .reshape([_stateSpaceWidth, _stateSpaceHeight])
                    .data<double>()
                    .ToNDArray();

                _heatmapPlots[i].UpdateHeatMapSeries(
                    conditionalIntensityValues
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

            if (_heatmapPlots != null)
            {
                for (int i = 0; i < _heatmapPlots.Count; i++)
                {
                    if (!_heatmapPlots[i].IsDisposed)
                    {
                        _heatmapPlots[i].Dispose();
                    }
                }
                _heatmapPlots = null;
            };

            _conditionalIntensitiesCount = 0;
            _conditionalIntensitiesCumulativeIndex.Clear();
            _conditionalIntensities = null;
        }

        public override IObservable<object> Visualize(IObservable<IObservable<object>> source, IServiceProvider provider)
        {
            if (provider.GetService(typeof(IDialogTypeVisualizerService)) is not Control visualizerControl)
            {
                return source;
            }

            return source.SelectMany(input => 
                input.Sample(TimeSpan.FromMilliseconds(_sampleFrequency))
                    .ObserveOn(visualizerControl)
                    .Do(value => 
                    {
                        var success = UpdateModel();
                        if (!success)
                        {
                            return;
                        }

                        var newConditionalIntensitiesCount = GetConditionalIntensitiesCount(_conditionalIntensities, _conditionalIntensitiesCumulativeIndex);
                        if (_conditionalIntensitiesCount != newConditionalIntensitiesCount)
                        {
                            _conditionalIntensitiesCount = newConditionalIntensitiesCount;
                            UpdatePages();
                            UpdateHeatmaps();
                            UpdateTableLayout();
                        }

                        Show(value);
                    }
                )
            );
        }
    }
}