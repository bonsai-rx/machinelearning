using System.Windows.Forms;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.WindowsForms;
using System.Drawing;
using System;
using OxyPlot.Axes;
using System.Collections.Generic;

namespace Bonsai.ML.Design
{
    /// <summary>
    /// Provides a user control to display 2D data as a heatmap using OxyPlot.
    /// </summary>
    public class HeatMapSeriesOxyPlotBase : UserControl
    {
        private PlotView _view;
        /// <summary>
        /// Gets the plot view of the control.
        /// </summary>
        public PlotView View => _view;

        private PlotModel _model;
        /// <summary>
        /// Gets the plot model of the control.
        /// </summary>
        public PlotModel Model => _model;

        private HeatMapSeries heatMapSeries;
        private LinearColorAxis colorAxis = null;

        private ToolStripComboBox paletteComboBox;
        private ToolStripLabel paletteLabel;
        private int _paletteSelectedIndex;
        private OxyPalette palette;

        private ToolStripComboBox renderMethodComboBox;
        private ToolStripLabel renderMethodLabel;
        private int _renderMethodSelectedIndex;
        private HeatMapRenderMethod renderMethod = HeatMapRenderMethod.Bitmap;
        private StatusStrip statusStrip;

        private ToolStripTextBox maxValueTextBox = null;
        private ToolStripLabel maxValueLabel;

        private ToolStripTextBox minValueTextBox = null;
        private ToolStripLabel minValueLabel;

        private ToolStripDropDownButton _visualizerPropertiesDropDown;
        /// <summary>
        /// Gets the visualizer properties drop down button.
        /// </summary>
        public ToolStripDropDownButton VisualizerPropertiesDropDown => _visualizerPropertiesDropDown;

        private int _numColors = 100;

        /// <summary>
        /// Event handler which can be used to hook into events generated when the combobox values have changed.
        /// </summary>
        public event EventHandler PaletteComboBoxValueChanged;

        /// <summary>
        /// Gets the palette combobox.
        /// </summary>
        public ToolStripComboBox PaletteComboBox => paletteComboBox;

        /// <summary>
        /// Event handler which can be used to hook into events generated when the render method combobox values have changed.
        /// </summary>
        public event EventHandler RenderMethodComboBoxValueChanged;

        /// <summary>
        /// Gets the render method combobox.
        /// </summary>
        public ToolStripComboBox RenderMethodComboBox => renderMethodComboBox;

        /// <summary>
        /// Gets the status strip control.
        /// </summary>
        public StatusStrip StatusStrip => statusStrip;

        private double? _valueMin = null;
        /// <summary>
        /// Gets or sets the minimum value of the color axis.
        /// </summary>
        public double? ValueMin 
        { 
            get => _valueMin;
            set
            {
                _valueMin = value;
                if (minValueTextBox != null)
                    minValueTextBox.Text = value?.ToString();
                if (colorAxis != null)
                    colorAxis.Maximum = value ?? double.NaN;
            }
        }

        private double? _valueMax = null;
        /// <summary>
        /// Gets or sets the maximum value of the color axis.
        /// </summary>
        public double? ValueMax
        {
            get => _valueMax;
            set
            {
                _valueMax = value;
                if (maxValueTextBox != null)
                    maxValueTextBox.Text = value?.ToString();
                if (colorAxis != null)
                    colorAxis.Maximum = value ?? double.NaN;
            }
        }

        /// <summary>
        /// Constructor of the TimeSeriesOxyPlotBase class.
        /// Requires a line series name and an area series name.
        /// Data source is optional, since pasing it to the constructor will populate the combobox and leave it empty otherwise.
        /// The selected index is only needed when the data source is provided.
        /// </summary>
        public HeatMapSeriesOxyPlotBase(
            int paletteSelectedIndex,
            int renderMethodSelectedIndex, 
            int numColors = 100
        )
        {
            _paletteSelectedIndex = paletteSelectedIndex;
            _renderMethodSelectedIndex = renderMethodSelectedIndex;
            _numColors = numColors;
            Initialize();
        }

        private void Initialize()
        {
            _view = new PlotView
            {
                Dock = DockStyle.Fill,
            };

            _model = new PlotModel();

            heatMapSeries = new HeatMapSeries {
                X0 = 0,
                X1 = 100,
                Y0 = 0,
                Y1 = 100,
                Interpolate = true,
                RenderMethod = renderMethod,
                CoordinateDefinition = HeatMapCoordinateDefinition.Edge
            };

            colorAxis = new LinearColorAxis {
                Position = AxisPosition.Right,
            };

            _model.Axes.Add(colorAxis);
            _model.Series.Add(heatMapSeries);

            _view.Model = _model;
            Controls.Add(_view);

            InitializeColorPalette();
            InitializeRenderMethod();
            InitializeColorAxisValues();

            statusStrip = new StatusStrip
            {
                Visible = false,
            };

            var toolStripItems = new ToolStripItem[] {
                paletteLabel,
                paletteComboBox,
                renderMethodLabel,
                renderMethodComboBox,
                maxValueLabel,
                maxValueTextBox,
                minValueLabel,
                minValueTextBox
            };

            _visualizerPropertiesDropDown = new ToolStripDropDownButton("Visualizer Properties");

            foreach (var item in toolStripItems)
            {
                _visualizerPropertiesDropDown.DropDownItems.Add(item);
            }

            statusStrip.Items.Add(_visualizerPropertiesDropDown);

            Controls.Add(statusStrip);
            _view.MouseClick += (sender, e) => {
                if (e.Button == MouseButtons.Right)
                {
                    statusStrip.Visible = !statusStrip.Visible;
                }
            };

            AutoScaleDimensions = new SizeF(6F, 13F);
        }

        private void InitializeColorAxisValues()
        {
            maxValueLabel = new ToolStripLabel
            {
                Text = "Maximum Value:",
                AutoSize = true
            };

            maxValueTextBox = new ToolStripTextBox()
            {
                Name = "maxValue",
                AutoSize = true,
                Text = _valueMax.HasValue ? _valueMax.ToString() : "auto",
            };

            maxValueTextBox.TextChanged += (sender, e) =>
            {
                if (double.TryParse(maxValueTextBox.Text, out double maxValue))
                {
                    colorAxis.Maximum = maxValue;
                    ValueMax = maxValue;
                }
                else if (maxValueTextBox.Text.ToLower() == "auto")
                {
                    colorAxis.Maximum = double.NaN;
                    maxValueTextBox.Text = "auto";
                    ValueMax = null;
                }
                else
                {
                    colorAxis.Maximum = heatMapSeries.MaxValue;
                    ValueMax = heatMapSeries.MaxValue;
                }
                UpdatePlot();
            };

            minValueLabel = new ToolStripLabel
            {
                Text = "Minimum Value:",
                AutoSize = true
            };

            minValueTextBox = new ToolStripTextBox()
            {
                Name = "minValue",
                AutoSize = true,
                Text = _valueMin.HasValue ? _valueMin.ToString() : "auto",
            };

            minValueTextBox.TextChanged += (sender, e) =>
            {
                if (double.TryParse(minValueTextBox.Text, out double minValue))
                {
                    colorAxis.Minimum = minValue;
                    ValueMin = minValue;
                }
                else if (minValueTextBox.Text.ToLower() == "auto")
                {
                    colorAxis.Minimum = double.NaN;
                    minValueTextBox.Text = "auto";
                    ValueMin = null;
                }
                else
                {
                    colorAxis.Minimum = heatMapSeries.MinValue;
                    ValueMin = heatMapSeries.MinValue;
                }
                UpdatePlot();
            };
        }

        private void InitializeColorPalette()
        {
            paletteLabel = new ToolStripLabel
            {
                Text = "Color palette:",
                AutoSize = true
            };

            paletteComboBox = new ToolStripComboBox()
            {
                Name = "palette"
            };

            foreach (var value in Enum.GetValues(typeof(ColorPalette)))
            {
                paletteComboBox.Items.Add(value);
            }

            paletteComboBox.SelectedIndexChanged += PaletteComboBoxSelectedIndexChanged;
            paletteComboBox.SelectedIndex = _paletteSelectedIndex;
            UpdateColorPalette();
        }

        private void PaletteComboBoxSelectedIndexChanged(object sender, EventArgs e)
        {
            if (paletteComboBox.SelectedIndex != _paletteSelectedIndex)
            {
                _paletteSelectedIndex = paletteComboBox.SelectedIndex;
                UpdateColorPalette();
                PaletteComboBoxValueChanged?.Invoke(this, e);
                UpdatePlot();
            }
        }

        private void UpdateColorPalette()
        {
            var selectedPalette = (ColorPalette)paletteComboBox.Items[_paletteSelectedIndex];
            paletteLookup.TryGetValue(selectedPalette, out Func<int, OxyPalette> paletteMethod);
            palette = paletteMethod(_numColors);
            colorAxis.Palette = palette;
        }

        private void InitializeRenderMethod()
        {
            renderMethodLabel = new ToolStripLabel
            {
                Text = "Render method:",
                AutoSize = true
            };

            renderMethodComboBox = new ToolStripComboBox()
            {
                Name = "renderMethod"
            };

            foreach (var value in Enum.GetValues(typeof(HeatMapRenderMethod)))
            {
                renderMethodComboBox.Items.Add(value);
            }

            renderMethodComboBox.SelectedIndexChanged += renderMethodComboBoxSelectedIndexChanged;
            renderMethodComboBox.SelectedIndex = _renderMethodSelectedIndex;
            UpdateRenderMethod();
        }

        private void renderMethodComboBoxSelectedIndexChanged(object sender, EventArgs e)
        {
            if (renderMethodComboBox.SelectedIndex != _renderMethodSelectedIndex)
            {
                _renderMethodSelectedIndex = renderMethodComboBox.SelectedIndex;
                UpdateRenderMethod();
                RenderMethodComboBoxValueChanged?.Invoke(this, e);
                UpdatePlot();
            }
        }

        private void UpdateRenderMethod()
        {
            renderMethod = (HeatMapRenderMethod)renderMethodComboBox.Items[_renderMethodSelectedIndex];
            heatMapSeries.RenderMethod = renderMethod;
        }

        /// <summary>
        /// Method to update the heatmap series with new data.
        /// </summary>
        /// <param name="data">The data to be displayed.</param>
        public void UpdateHeatMapSeries(double[,] data)
        {
            heatMapSeries.Data = data;
        }


        /// <summary>
        /// Method to update the heatmap series X axis range.
        /// </summary>
        /// <param name="x0"></param>
        /// <param name="x1"></param>
        public void UpdateXRange(double x0, double x1)
        {
            heatMapSeries.X0 = x0;
            heatMapSeries.X1 = x1;
        }

        /// <summary>
        /// Method to update the heatmap series Y axis range.
        /// </summary>
        /// <param name="y0"></param>
        /// <param name="y1"></param>
        public void UpdateYRange(double y0, double y1)
        {
            heatMapSeries.Y0 = y0;
            heatMapSeries.Y1 = y1;
        }

        /// <summary>
        /// Method to update the heatmap series with new data.
        /// </summary>
        /// <param name="x0">The minimum x value.</param>
        /// <param name="x1">The maximum x value.</param>
        /// <param name="y0">The minimum y value.</param>
        /// <param name="y1">The maximum y value.</param>
        /// <param name="data">The data to be displayed.</param>
        public void UpdateHeatMapSeries(double x0, double x1, double y0, double y1, double[,] data)
        {
            UpdateXRange(x0, x1);
            UpdateYRange(y0, y1);
            UpdateHeatMapSeries(data);
        }

        /// <summary>
        /// Method to update the plot.
        /// </summary>
        public void UpdatePlot()
        {
            _model.InvalidatePlot(true);
        }

        private static readonly Dictionary<ColorPalette, Func<int, OxyPalette>> paletteLookup = new Dictionary<ColorPalette, Func<int, OxyPalette>>
        {
            { ColorPalette.Cividis, OxyPalettes.Cividis },
            { ColorPalette.Inferno, OxyPalettes.Inferno },
            { ColorPalette.Viridis, OxyPalettes.Viridis },
            { ColorPalette.Magma, OxyPalettes.Magma },
            { ColorPalette.Plasma, OxyPalettes.Plasma },
            { ColorPalette.BlackWhiteRed, OxyPalettes.BlackWhiteRed },
            { ColorPalette.BlueWhiteRed, OxyPalettes.BlueWhiteRed },
            { ColorPalette.Cool, OxyPalettes.Cool },
            { ColorPalette.Gray, OxyPalettes.Gray },
            { ColorPalette.Hot, OxyPalettes.Hot },
            { ColorPalette.Hue, OxyPalettes.Hue },
            { ColorPalette.HueDistinct, OxyPalettes.HueDistinct },
            { ColorPalette.Jet, OxyPalettes.Jet },
            { ColorPalette.Rainbow, OxyPalettes.Rainbow },
        };
    }
}
