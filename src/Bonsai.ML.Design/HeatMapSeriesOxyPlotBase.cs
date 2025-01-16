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
        private PlotView view;
        private PlotModel model;
        private HeatMapSeries heatMapSeries;
        private LinearColorAxis colorAxis;

        private ToolStripComboBox paletteComboBox;
        private ToolStripLabel paletteLabel;
        private int _paletteSelectedIndex;
        private OxyPalette palette;

        private ToolStripComboBox renderMethodComboBox;
        private ToolStripLabel renderMethodLabel;
        private int _renderMethodSelectedIndex;
        private HeatMapRenderMethod renderMethod = HeatMapRenderMethod.Bitmap;
        private StatusStrip statusStrip;

        private ToolStripTextBox maxValueTextBox;
        private ToolStripLabel maxValueLabel;

        private ToolStripTextBox minValueTextBox;
        private ToolStripLabel minValueLabel;

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

        /// <summary>
        /// Gets the plot model.
        /// </summary>
        public PlotModel Model => model;

        /// <summary>
        /// Constructor of the TimeSeriesOxyPlotBase class.
        /// Requires a line series name and an area series name.
        /// Data source is optional, since pasing it to the constructor will populate the combobox and leave it empty otherwise.
        /// The selected index is only needed when the data source is provided.
        /// </summary>
        public HeatMapSeriesOxyPlotBase(int paletteSelectedIndex, int renderMethodSelectedIndex, int numColors = 100)
        {
            _paletteSelectedIndex = paletteSelectedIndex;
            _renderMethodSelectedIndex = renderMethodSelectedIndex;
            _numColors = numColors;
            Initialize();
        }

        private void Initialize()
        {
            view = new PlotView
            {
                Dock = DockStyle.Fill,
            };

            model = new PlotModel();

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

            model.Axes.Add(colorAxis);
            model.Series.Add(heatMapSeries);

            view.Model = model;
            Controls.Add(view);

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

            ToolStripDropDownButton visualizerPropertiesButton = new ToolStripDropDownButton("Visualizer Properties");

            foreach (var item in toolStripItems)
            {
                visualizerPropertiesButton.DropDownItems.Add(item);
            }

            statusStrip.Items.Add(visualizerPropertiesButton);

            Controls.Add(statusStrip);
            view.MouseClick += new MouseEventHandler(onMouseClick);
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
                Text = "auto",
            };

            maxValueTextBox.TextChanged += (sender, e) =>
            {
                if (double.TryParse(maxValueTextBox.Text, out double maxValue))
                {
                    colorAxis.Maximum = maxValue;
                }
                else if (maxValueTextBox.Text.ToLower() == "auto")
                {
                    colorAxis.Maximum = double.NaN;
                    maxValueTextBox.Text = "auto";
                }
                else
                {
                    colorAxis.Maximum = heatMapSeries.MaxValue;
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
                Text = "auto",
            };

            minValueTextBox.TextChanged += (sender, e) =>
            {
                if (double.TryParse(minValueTextBox.Text, out double minValue))
                {
                    colorAxis.Minimum = minValue;
                }
                else if (minValueTextBox.Text.ToLower() == "auto")
                {
                    colorAxis.Minimum = double.NaN;
                    minValueTextBox.Text = "auto";
                }
                else
                {
                    colorAxis.Minimum = heatMapSeries.MinValue;
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

        private void onMouseClick(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Right)
            {
                statusStrip.Visible = !statusStrip.Visible;
            }
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
        /// Method to update the heatmap x axis.
        /// </summary>
        /// <param name="x0"></param>
        /// <param name="x1"></param>
        public void UpdateHeatMapXAxis(double x0, double x1)
        {
            heatMapSeries.X0 = x0;
            heatMapSeries.X1 = x1;
        }

        /// <summary>
        /// Method to update the heatmap y axis.
        /// </summary>
        /// <param name="y0"></param>
        /// <param name="y1"></param>
        public void UpdateHeatMapYAxis(double y0, double y1)
        {
            heatMapSeries.Y0 = y0;
            heatMapSeries.Y1 = y1;
        }

        /// <summary>
        /// Method to update the heatmap axes.
        /// </summary>
        /// <param name="x0"></param>
        /// <param name="x1"></param>
        /// <param name="y0"></param>
        /// <param name="y1"></param>
        public void UpdateHeatMapAxes(double x0, double x1, double y0, double y1)
        {
            UpdateHeatMapXAxis(x0, x1);
            UpdateHeatMapYAxis(y0, y1);
        }

        /// <summary>
        /// Method to update the heatmap series data and axes.
        /// </summary>
        /// <param name="x0">The minimum x value.</param>
        /// <param name="x1">The maximum x value.</param>
        /// <param name="y0">The minimum y value.</param>
        /// <param name="y1">The maximum y value.</param>
        /// <param name="data">The data to be displayed.</param>
        public void UpdateHeatMapSeries(double x0, double x1, double y0, double y1, double[,] data)
        {
            UpdateHeatMapAxes(x0, x1, y0, y1);
            heatMapSeries.Data = data;
        }

        /// <summary>
        /// Method to update the plot.
        /// </summary>
        public void UpdatePlot()
        {
            model.InvalidatePlot(true);
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
