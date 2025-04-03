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

[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.SelectUnitIdsVisualizer), 
    Target = typeof(Bonsai.ML.PointProcessDecoder.SelectUnitIds))]

namespace Bonsai.ML.PointProcessDecoder.Design
{
    /// <summary>
    /// Visualizer for the <see cref="SelectUnitIds"/> node.
    /// </summary>
    public class SelectUnitIdsVisualizer : DialogTypeVisualizer
    {
        private int _rowCount = 1;
        /// <summary>
        /// The number of rows in the visualizer.
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
        /// The number of columns in the visualizer.
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
        /// The index of the current page displayed in the visualizer.
        /// </summary>
        public int SelectedPageIndex
        {
            get => _selectedPageIndex;
            set
            {
                if (value < 0)
                {
                    throw new InvalidOperationException("The selected page index must be greater than or equal to 0.");
                }
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
        private readonly int _sampleFrequency = 30;
        private int _pageCount = 1;
        private string _modelName = string.Empty;
        private List<HeatMapSeriesOxyPlotBase> _heatmapPlots = null;
        private int _intensitiesCount = 0;
        private TableLayoutPanel _container = null;
        private readonly List<long> _intensitiesCumulativeIndex = [];
        private ToolStripNumericUpDown _pageIndexControl = null;
        private ToolStripNumericUpDown _rowControl = null;
        private ToolStripNumericUpDown _columnControl = null;

        private ToolStripButton _displaySelectedButton = null;
        private ToolStripButton _resetSelectedButton = null;

        private Tensor[] _intensities = null;
        private long _stateSpaceWidth;
        private long _stateSpaceHeight;
        private double[] _stateSpaceMin;
        private double[] _stateSpaceMax;
        private bool _isProcessing = false;

        private SelectUnitIds _node = null;
        private Tensor _selectedUnits = empty(0, dtype: ScalarType.Int32);

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            var expressionBuilderGraph = (ExpressionBuilderGraph)provider.GetService(typeof(ExpressionBuilderGraph));
            var typeVisualizerContext = (ITypeVisualizerContext)provider.GetService(typeof(ITypeVisualizerContext));
            if (expressionBuilderGraph != null && typeVisualizerContext != null)
            {
                _node = ExpressionBuilder.GetWorkflowElement(
                    expressionBuilderGraph.Where(node => node.Value == typeVisualizerContext.Source)
                        .FirstOrDefault().Value) as SelectUnitIds;
            }

            if (_node == null)
            {
                throw new InvalidOperationException("The decode node is invalid.");
            }

            _modelName = _node.Name;
            if (string.IsNullOrEmpty(_modelName))
            {
                throw new InvalidOperationException("The point process model name is not set.");
            }

            _selectedUnits = _node.UnitIds;
            
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
                    UpdateTableLayout();
                    Show(null);
                }
            };

            _resetSelectedButton = new ToolStripButton("Reset Selection");

            _resetSelectedButton.Click += (sender, e) =>
            {
                if (!_displaySelected)
                {
                    for (int i = 0; i < _selectedUnits.size(0); i++)
                    {
                        var index = _selectedUnits[i].item<int>();
                        _heatmapPlots[index].View.BackColor = SystemColors.Control;
                    }
                }

                _selectedUnits = empty(0, dtype: ScalarType.Int32);
                _node.UnitIds = _selectedUnits;

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
                (int)Math.Ceiling((double)_selectedUnits.size(0) / (_rowCount * _columnCount)) : 
                (int)Math.Ceiling((double)_intensitiesCount / (_rowCount * _columnCount));
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
                throw new InvalidOperationException("For the intensities visualizer to work, the state space dimensions must be 2.");
            }

            if (model.Encoder.Intensities.Length == 0 || (model.Encoder.Intensities.Length == 1 && model.Encoder.Intensities[0].NumberOfElements == 0))
            {
                return false;
            }
            
            _intensities = model.Encoder.Intensities;
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

            _isProcessing = false;

            return true;
        }

        private static int GetIntensitiesCount(Tensor[] intensities, List<long> intensitiesCumulativeIndex)
        {
            long intensitiesCount = 0;
            intensitiesCumulativeIndex.Clear();
            for (int i = 0; i < intensities.Length; i++) {
                if (intensities[i].NumberOfElements > 0) {
                    intensitiesCount += intensities[i].size(0);
                    intensitiesCumulativeIndex.Add(intensitiesCount);
                }
            }
            return (int)intensitiesCount;
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
                    var containsIndex = (_selectedUnits == index).any().item<bool>();

                    if (containsIndex)
                    {
                        if (_selectedUnits.NumberOfElements == 1)
                        {
                            _selectedUnits = empty(0, dtype: ScalarType.Int32);
                        }
                        else
                        {
                            _selectedUnits = _selectedUnits[_selectedUnits != index];
                        }
                        control.BackColor = SystemColors.Control;
                    }
                    else
                    {
                        var indexTensor = tensor(new int[] { index }, dtype: ScalarType.Int32);
                        if (_selectedUnits.NumberOfElements == 0)
                        {
                            _selectedUnits = indexTensor;
                        }
                        else
                        {
                            _selectedUnits = cat([_selectedUnits, indexTensor], dim: 0);
                        }
                        control.BackColor = Color.LightBlue;
                    }
                    _node.UnitIds = _selectedUnits;
                }
            };
        }

        private bool UpdateHeatmaps()
        {
            if (_heatmapPlots is null)
            {
                _heatmapPlots = [];
                for (int i = 0; i < _intensitiesCount; i++)
                {
                    AddHeatmap();
                }
            }
            else if (_heatmapPlots.Count > _intensitiesCount)
            {
                var count = _heatmapPlots.Count - _intensitiesCount;
                for (int i = 0; i < count; i++)
                {
                    if (!_heatmapPlots[i + _intensitiesCount].IsDisposed)
                    {
                        _heatmapPlots[i + _intensitiesCount].Dispose();
                    }
                }
                _heatmapPlots.RemoveRange(_intensitiesCount, count);
            }
            else if (_heatmapPlots.Count < _intensitiesCount)
            {
                for (int i = _heatmapPlots.Count; i < _intensitiesCount; i++)
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
                        if (selectionIndex >= _selectedUnits.size(0))
                        {
                            break;
                        }
                        
                        index = _selectedUnits[selectionIndex].item<int>();
                        _heatmapPlots[index].View.BackColor = SystemColors.Control;
                    }
                    else 
                    {
                        index = SelectedPageIndex * _rowCount * _columnCount + i * _columnCount + j;
                        if (index >= _intensitiesCount)
                        {
                            break;
                        }
                        var containsIndex = (_selectedUnits == index).any().item<bool>();
                        if (containsIndex)
                        {
                            _heatmapPlots[index].View.BackColor = Color.LightBlue;
                        }
                    }

                    _container.Controls.Add(_heatmapPlots[index], j, i);
                }
            }
        }

        private (int intensitiesIndex, int intensitiesTensorIndex) GetIntensitiesIndex(int index)
        {

            var intensitiesIndex = 0;
            for (int i = 0; i < _intensitiesCumulativeIndex.Count; i++)
            {
                if (index < _intensitiesCumulativeIndex[i])
                {
                    intensitiesIndex = i;
                    break;
                }
            }
            var intensitiesTensorIndex = intensitiesIndex == 0 ? index : index - _intensitiesCumulativeIndex[intensitiesIndex - 1];
            return (intensitiesIndex, (int)intensitiesTensorIndex);
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            var startIndex = SelectedPageIndex * _rowCount * _columnCount;
            var endIndex = _displaySelected ? 
                Math.Min(startIndex + _rowCount * _columnCount, (int)_selectedUnits.size(0)) : 
                Math.Min(startIndex + _rowCount * _columnCount, _intensitiesCount);

            for (int i = startIndex; i < endIndex; i++) 
            {
                if (_displaySelected)
                {
                    i = _selectedUnits[i].item<int>();
                }

                var (intensitiesIndex, intensitiesTensorIndex) = GetIntensitiesIndex(i);
                var intensity = _intensities[intensitiesIndex][intensitiesTensorIndex];

                if (intensity.Dimensions == 2) {
                    intensity = intensity
                        .sum(dim: 0);
                }

                var intensityValues = (double[,])intensity
                    .exp()
                    .to_type(ScalarType.Float64)
                    .reshape([_stateSpaceWidth, _stateSpaceHeight])
                    .data<double>()
                    .ToNDArray();

                _heatmapPlots[i].UpdateHeatMapSeries(
                    _stateSpaceMin[0],
                    _stateSpaceMax[0],
                    _stateSpaceMin[1],
                    _stateSpaceMax[1],
                    intensityValues
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

            _intensitiesCount = 0;
            _intensitiesCumulativeIndex.Clear();
            _intensities = null;
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
                        if (!UpdateModel() && !_isProcessing)
                        {
                            return;
                        }

                        var newIntensitiesCount = GetIntensitiesCount(_intensities, _intensitiesCumulativeIndex);
                        if (_intensitiesCount != newIntensitiesCount)
                        {
                            _intensitiesCount = newIntensitiesCount;
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