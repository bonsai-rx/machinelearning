using System;
using System.Reactive.Linq;
using System.ComponentModel;
using System.Drawing;
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
    Target = typeof(Bonsai.ML.PointProcessDecoder.CreatePointProcessModel))]
[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.DensityEstimationsVisualizer), 
    Target = typeof(Bonsai.ML.PointProcessDecoder.Decode))]
[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.DensityEstimationsVisualizer), 
    Target = typeof(Bonsai.ML.PointProcessDecoder.Encode))]
[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.DensityEstimationsVisualizer), 
    Target = typeof(Bonsai.ML.PointProcessDecoder.GetModel))]
[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.DensityEstimationsVisualizer), 
    Target = typeof(Bonsai.ML.PointProcessDecoder.LoadPointProcessModel))]
[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.DensityEstimationsVisualizer), 
    Target = typeof(Bonsai.ML.PointProcessDecoder.SavePointProcessModel))]

namespace Bonsai.ML.PointProcessDecoder.Design
{
    /// <summary>
    /// Visualizer for the density estimations of a point process model.
    /// </summary>
    public class DensityEstimationsVisualizer : DialogTypeVisualizer
    {
        private int _rowCount = 1;
        /// <summary>
        /// The number of rows in the table layout.
        /// </summary>
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
        /// <summary>
        /// The number of columns in the table layout.
        /// </summary>
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
        /// <summary>
        /// The index of the selected page in the table layout.
        /// </summary>
        public int SelectedPageIndex
        {
            get => _selectedPageIndex;
            set
            {
                _selectedPageIndex = value;
            }
        }

        private StatusStrip _statusStrip = null;
        /// <summary>
        /// The status strip control that displays the visualizer options.
        /// </summary>
        public StatusStrip StatusStrip => _statusStrip;

        private bool _displaySelected = false;
        /// <summary>
        /// Gets or sets a value indicating whether to display only the selected cells.
        /// </summary>
        public bool DisplaySelected
        {
            get => _displaySelected;
            set
            {
                _displaySelected = value;
            }
        }

        private List<int> _selectedCells = [];
        /// <summary>
        /// Gets or sets the selected cells in the visualizer.
        /// </summary>
        [TypeConverter(typeof(UnidimensionalArrayConverter))]
        public int[] SelectedCells
        {
            get => [.. _selectedCells];
            set
            {
                _selectedCells = [.. value];
            }
        }

        private readonly int _sampleFrequency = 200;
        private int _pageCount = 1;
        private string _modelName = string.Empty;
        private List<HeatMapSeriesOxyPlotBase> _heatmapPlots = null;
        private int _estimationsCount = 0;
        private TableLayoutPanel _container = null;
        private ToolStripNumericUpDown _pageIndexControl = null;
        private ToolStripNumericUpDown _rowControl = null;
        private ToolStripNumericUpDown _columnControl = null;
        private ToolStripButton _displaySelectedButton = null;
        private ToolStripButton _resetSelectedButton = null;
        private Tensor[] _estimations = null;
        private long _stateSpaceWidth;
        private long _stateSpaceHeight;
        private double[] _stateSpaceMin = null;
        private double[] _stateSpaceMax = null;
        private bool _isProcessing = false;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            IManagedPointProcessModelNode node = null;
            var expressionBuilderGraph = (ExpressionBuilderGraph)provider.GetService(typeof(ExpressionBuilderGraph));
            var typeVisualizerContext = (ITypeVisualizerContext)provider.GetService(typeof(ITypeVisualizerContext));
            if (expressionBuilderGraph != null && typeVisualizerContext != null)
            {
                var element = typeVisualizerContext.Source.Builder;
                
                if (element != null)
                {
                    node = ExpressionBuilder.GetWorkflowElement(element) as IManagedPointProcessModelNode;
                }
            }

            if (node == null)
            {
                throw new InvalidOperationException("Unable to access visualizer's source node.");
            }

            _modelName = node.Name;
            if (string.IsNullOrEmpty(_modelName))
            {
                throw new InvalidOperationException("The point process model name is not set.");
            }
            
            _container = new TableLayoutPanel()
            {
                Dock = DockStyle.Fill,
                AutoSize = true,
                ColumnCount = _columnCount,
                RowCount = _rowCount,
            };

            var pageIndexLabel = new ToolStripLabel($"Page: ");
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
                
                if (_isProcessing || !UpdateModel())
                {
                    return;
                }

                UpdateTableLayout();
                Show(null);
            };

            var rowLabel = new ToolStripLabel($"Rows: ");
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
                    if (_isProcessing || !UpdateModel())
                    {
                        return;
                    }
                    
                    UpdateTableLayout();
                    Show(null);
                }
            };

            var columnLabel = new ToolStripLabel($"Columns: ");
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
                    if (_isProcessing || !UpdateModel())
                    {
                        return;
                    }

                    UpdateTableLayout();
                    Show(null);
                }
            };

            _displaySelectedButton = new ToolStripButton("Display Selection")
            {
                CheckOnClick = true,
                Checked = _displaySelected,
            };

            _displaySelectedButton.CheckedChanged += (sender, e) =>
            {
                if (_heatmapPlots is null)
                {
                    return;
                }

                _displaySelected = _displaySelectedButton.Checked;

                UpdatePages();

                if (_selectedPageIndex >= _pageCount) 
                {
                    SelectedPageIndex = _pageCount - 1;
                    _pageIndexControl.Value = _selectedPageIndex;
                }
                else 
                {
                    if (_isProcessing || !UpdateModel())
                    {
                        return;
                    }
                    
                    UpdateTableLayout();
                    Show(null);
                }
            };

            _resetSelectedButton = new ToolStripButton("Reset Selection");

            _resetSelectedButton.Click += (sender, e) =>
            {
                if (!_displaySelected)
                {
                    for (int i = 0; i < _selectedCells.Count; i++)
                    {
                        _heatmapPlots[_selectedCells[i]].View.BackColor = SystemColors.Control;
                    }
                }

                _selectedCells.Clear();
                _displaySelected = false;
                if (_displaySelectedButton.Checked)
                {
                    _displaySelectedButton.Checked = false;
                }
                else
                {
                    if (_heatmapPlots is null)
                    {
                        return;
                    }

                    UpdatePages();

                    if (_selectedPageIndex >= _pageCount) 
                    {
                        SelectedPageIndex = _pageCount - 1;
                        _pageIndexControl.Value = _selectedPageIndex;
                    }
                    else 
                    {
                        if (_isProcessing || !UpdateModel())
                        {
                            return;
                        }

                        UpdateTableLayout();
                        Show(null);
                    }
                }
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
                _columnControl,
                _displaySelectedButton,
                _resetSelectedButton
            ]);

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            visualizerService?.AddControl(_container);
            visualizerService?.AddControl(_statusStrip);
        }

        private void UpdatePages()
        {
            _pageCount = _displaySelected ? 
                (int)Math.Ceiling((double)_selectedCells.Count / (_rowCount * _columnCount)) : 
                (int)Math.Ceiling((double)_estimationsCount / (_rowCount * _columnCount));
            _pageCount = Math.Max(_pageCount, 1);
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

            var estimationsCount = model.Encoder.Estimations.Length;

            if (estimationsCount == 0)
            {
                return false;
            }
            
            if (_estimations is null || estimationsCount != _estimations.Length)
                _estimations = new Tensor[estimationsCount];

            var startIndex = _selectedPageIndex * _rowCount * _columnCount;
            var endIndex = _displaySelected ? 
                Math.Min(startIndex + _rowCount * _columnCount, _selectedCells.Count) : 
                Math.Min(startIndex + _rowCount * _columnCount, estimationsCount);

            for (int i = startIndex; i < endIndex; i++)
            {
                var index = i;
                if (_displaySelected)
                {
                    index = _selectedCells[i];
                }

                var estimate = model.Encoder.Estimations[index].Estimate(
                    model.StateSpace.Points,
                    null,
                    model.StateSpace.Dimensions
                );

                if (estimate.NumberOfElements == 0) {
                    _estimations[index] = ones([model.StateSpace.Points.size(0), 1]) * double.NaN;
                } else {
                    _estimations[index] = model.Encoder.Estimations[index].Normalize(estimate);
                }
            }

            _stateSpaceWidth = model.StateSpace.Shape[0];
            _stateSpaceHeight = model.StateSpace.Shape[1];

            _stateSpaceMin ??= [.. model.StateSpace.Points
                .min(dim: 0)
                .values
                .to_type(ScalarType.Float64)
                .data<double>()
            ];
            _stateSpaceMax ??= [.. model.StateSpace.Points
                .max(dim: 0)
                .values
                .to_type(ScalarType.Float64)
                .data<double>()
            ];

            _isProcessing = false;

            return true;
        }

        private void AddHeatmap()
        {
            var heatmap = new HeatMapSeriesOxyPlotBase(1, 0)
            {
                Dock = DockStyle.Fill,
            };

            _heatmapPlots.Add(heatmap);
            heatmap.View.Tag = _heatmapPlots.Count - 1;

            heatmap.View.MouseDown += (sender, eventArgs) =>
            {
                if (eventArgs.Button != MouseButtons.Left)
                {
                    return;
                }

                if ((Control.ModifierKeys & Keys.Control) == 0)
                {
                    return;
                }
                
                var control = (Control)sender;

                if (sender != null)
                {
                    var index = (int)control.Tag;

                    if (_selectedCells.Contains(index))
                    {
                        _selectedCells.Remove(index);
                        control.BackColor = SystemColors.Control;
                    }
                    else
                    {
                        _selectedCells.Add(index);
                        control.BackColor = Color.LightBlue;
                    }
                }
            };
        }

        private bool UpdateHeatmaps()
        {
            if (_heatmapPlots is null)
            {
                _heatmapPlots = [];
                for (int i = 0; i < _estimationsCount; i++)
                {
                    AddHeatmap();
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
                    AddHeatmap();
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
                    int index;
                    if (_displaySelected)
                    {
                        var selectionIndex = i * _columnCount + j;
                        if (selectionIndex >= _selectedCells.Count)
                        {
                            break;
                        }
                        
                        index = _selectedCells[selectionIndex];
                        _heatmapPlots[index].View.BackColor = SystemColors.Control;
                    }
                    else 
                    {
                        index = _selectedPageIndex * _rowCount * _columnCount + i * _columnCount + j;
                        if (index >= _estimationsCount)
                        {
                            break;
                        }
                        if (_selectedCells.Contains(index))
                        {
                            _heatmapPlots[index].View.BackColor = Color.LightBlue;
                        }
                    }

                    _container.Controls.Add(_heatmapPlots[index], j, i);
                }
            }
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            var startIndex = _selectedPageIndex * _rowCount * _columnCount;
            var endIndex = _displaySelected ? 
                Math.Min(startIndex + _rowCount * _columnCount, _selectedCells.Count) : 
                Math.Min(startIndex + _rowCount * _columnCount, _estimationsCount);

            for (int i = startIndex; i < endIndex; i++) 
            {
                if (_displaySelected)
                {
                    i = _selectedCells[i];
                }

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
            _selectedCells.Clear();
        }

        /// <inheritdoc/>
        public override IObservable<object> Visualize(IObservable<IObservable<object>> source, IServiceProvider provider)
        {
            if (provider.GetService(typeof(IDialogTypeVisualizerService)) is not Control visualizerControl)
            {
                return source;
            }

            return source.SelectMany(input => 
                input.Sample(TimeSpan.FromMilliseconds(_sampleFrequency), HighResolutionScheduler.Default)
                    .ObserveOn(visualizerControl)
                    .Do(value => 
                    {
                        if (_isProcessing || !UpdateModel())
                        {
                            return;
                        }

                        if (_estimations.Length != _estimationsCount)
                        {
                            _estimationsCount = _estimations.Length;
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