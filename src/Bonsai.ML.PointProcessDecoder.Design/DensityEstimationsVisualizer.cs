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

[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.DensityEstimationsVisualizer), 
    Target = typeof(Bonsai.ML.PointProcessDecoder.Decode))]

namespace Bonsai.ML.PointProcessDecoder.Design
{
    public class DensityEstimationsVisualizer : DialogTypeVisualizer
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

        private readonly int _sampleFrequency = 5;
        private int _pageCount = 1;
        private string _modelName = string.Empty;
        private List<HeatMapSeriesOxyPlotBase> _heatmapPlots = null;
        private int _estimationsCount = 0;
        private TableLayoutPanel _container = null;
        private StatusStrip _statusStrip = null;
        public StatusStrip StatusStrip => _statusStrip;
        private ToolStripNumericUpDown _pageIndexControl = null;
        private ToolStripNumericUpDown _rowControl = null;
        private ToolStripNumericUpDown _columnControl = null;
        private Tensor[] _estimations = null;
        private long _stateSpaceWidth;
        private long _stateSpaceHeight;
        private double[] _stateSpaceMin;
        private double[] _stateSpaceMax;
        private bool _isProcessing = false;

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
                throw new InvalidOperationException("The decode node is invalid.");
            }

            _modelName = decodeNode.Model;
            if (string.IsNullOrEmpty(_modelName))
            {
                throw new InvalidOperationException("The point process model name is not set.");
            }
            
            _container = new TableLayoutPanel()
            {
                Dock = DockStyle.Fill,
                AutoSize = true,
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
                if (_heatmapPlots is null)
                {
                    return;
                }
                
                var value = Convert.ToInt32(_pageIndexControl.Value);
                SelectedPageIndex = value;
                UpdateTableLayout();
                if (UpdateModel())
                {
                    Show(null);
                }
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
                if (_heatmapPlots is null)
                {
                    return;
                }

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
                    if (UpdateModel())
                    {
                        Show(null);
                    }
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
                if (_heatmapPlots is null)
                {
                    return;
                }

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
                    if (UpdateModel())
                    {
                        Show(null);
                    }
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
            _pageCount = (int)Math.Ceiling((double)_estimationsCount / (_rowCount * _columnCount));
            _pageIndexControl.Maximum = _pageCount - 1;
        }

        private bool UpdateModel()
        {
            _isProcessing = true;
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

            if (model.Encoder.Estimations.Length == 0)
            {
                return false;
            }


            _estimationsCount = model.Encoder.Estimations.Length;
            _estimations = new Tensor[_estimationsCount];

            var startIndex = SelectedPageIndex * _rowCount * _columnCount;
            var endIndex = Math.Min(startIndex + _rowCount * _columnCount, _estimationsCount);

            for (int i = startIndex; i < endIndex; i++)
            {
                var estimate = model.Encoder.Estimations[i].Estimate(
                    model.StateSpace.Points,
                    null,
                    model.StateSpace.Dimensions
                );

                if (estimate.NumberOfElements == 0) {
                    _estimations[i] = ones([model.StateSpace.Points.size(0), 1]) * double.NaN;
                } else {
                    _estimations[i] = model.Encoder.Estimations[i].Normalize(estimate);
                }
            }

            _stateSpaceWidth = model.StateSpace.Shape[0];
            _stateSpaceHeight = model.StateSpace.Shape[1];

            _stateSpaceMin = [.. model.StateSpace.Points
                .min(dim: 0)
                .values
                .to_type(ScalarType.Float64)
                .data<double>()
            ];
            _stateSpaceMax = [.. model.StateSpace.Points
                .max(dim: 0)
                .values
                .to_type(ScalarType.Float64)
                .data<double>()
            ];

            // GC.KeepAlive(model);
            _isProcessing = false;

            return true;
        }

        private bool UpdateHeatmaps()
        {
            if (_heatmapPlots is null)
            {
                _heatmapPlots = [];
                for (int i = 0; i < _estimationsCount; i++)
                {
                    _heatmapPlots.Add(new HeatMapSeriesOxyPlotBase(1, 0)
                    {
                        Dock = DockStyle.Fill,
                    });
                }
            }
            else if (_heatmapPlots.Count > _estimationsCount)
            {
                var count = _heatmapPlots.Count - _estimationsCount;
                for (int i = 0; i < count; i++)
                {
                    if (!_heatmapPlots[i + _estimationsCount].IsDisposed)
                    {
                        _heatmapPlots[i + _estimationsCount].Dispose();
                    }
                }
                _heatmapPlots.RemoveRange(_estimationsCount, count);
            }
            else if (_heatmapPlots.Count < _estimationsCount)
            {
                for (int i = _heatmapPlots.Count; i < _estimationsCount; i++)
                {
                    _heatmapPlots.Add(new HeatMapSeriesOxyPlotBase(1, 0)
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
                    if (index >= _estimationsCount)
                    {
                        break;
                    }

                    _container.Controls.Add(_heatmapPlots[index], j, i);
                }
            }
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            var startIndex = SelectedPageIndex * _rowCount * _columnCount;
            var endIndex = Math.Min(startIndex + _rowCount * _columnCount, _estimationsCount);

            for (int i = startIndex; i < endIndex; i++) 
            {
                var estimation = _estimations[i];

                var estimationValues = (double[,])estimation
                    .to_type(ScalarType.Float64)
                    .reshape([_stateSpaceWidth, _stateSpaceHeight])
                    .data<double>()
                    .ToNDArray();

                _heatmapPlots[i].UpdateHeatMapSeries(
                    _stateSpaceMin[0],
                    _stateSpaceMax[0],
                    _stateSpaceMin[1],
                    _stateSpaceMax[1],
                    estimationValues
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

            _estimationsCount = 0;
            _estimations = null;
        }

        public override IObservable<object> Visualize(IObservable<IObservable<object>> source, IServiceProvider provider)
        {
            if (provider.GetService(typeof(IDialogTypeVisualizerService)) is not Control visualizerControl)
            {
                return source;
            }

            var timer = Observable.Interval(
                TimeSpan.FromMilliseconds(100),
                HighResolutionScheduler.Default
            );

            return source.SelectMany(input => 
                input.Buffer(timer)
                    .Where(buffer => buffer.Count > 0 && !_isProcessing)
                    .ObserveOn(visualizerControl)
                    .Do(buffer => 
                    {
                        if (!UpdateModel())
                        {
                            return;
                        }

                        UpdatePages();
                        UpdateHeatmaps();
                        UpdateTableLayout();

                        Show(buffer.LastOrDefault());
                    }
                )
            );
        }
    }
}