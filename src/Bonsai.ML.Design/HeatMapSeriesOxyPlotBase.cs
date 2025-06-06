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
        private HeatMapSeries heatMapSeries;
        private LinearColorAxis colorAxis = null;
        private ToolStripComboBox paletteComboBox;
        private ToolStripLabel paletteLabel;
        private int paletteSelectedIndex;
        private OxyPalette palette;
        private ToolStripComboBox renderMethodComboBox;
        private ToolStripLabel renderMethodLabel;
        private int renderMethodSelectedIndex;
        private HeatMapRenderMethod renderMethod = HeatMapRenderMethod.Bitmap;
        private StatusStrip statusStrip;
        private ToolStripTextBox maxValueTextBox = null;
        private ToolStripLabel maxValueLabel;
        private ToolStripTextBox minValueTextBox = null;
        private ToolStripLabel minValueLabel;

        /// <summary>
        /// Gets the visualizer properties drop down button.
        /// </summary>
        public ToolStripDropDownButton VisualizerPropertiesDropDown { get; private set; }

        /// <summary>
        /// Gets the plot view of the control.
        /// </summary>
        public PlotView View { get; private set; }

        /// <summary>
        /// Gets the plot model of the control.
        /// </summary>
        public PlotModel Model { get; private set; }

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
        /// Gets or sets the minimum value of the color axis.
        /// </summary>
        public double? ValueMin { get; set; }

        /// <summary>
        /// Gets or sets the maximum value of the color axis.
        /// </summary>
        public double? ValueMax { get; set; }

        /// <summary>
        /// Constructor of the TimeSeriesOxyPlotBase class.
        /// Requires a line series name and an area series name.
        /// Data source is optional, since pasing it to the constructor will populate the combobox and leave it empty otherwise.
        /// The selected index is only needed when the data source is provided.
        /// </summary>
        public HeatMapSeriesOxyPlotBase(
            int paletteSelectedIndex,
            int renderMethodSelectedIndex,
            double? valueMin = null,
            double? valueMax = null
        )
        {
            this.paletteSelectedIndex = paletteSelectedIndex;
            this.renderMethodSelectedIndex = renderMethodSelectedIndex;
            ValueMin = valueMin;
            ValueMax = valueMax;
            Initialize();
        }

        private void Initialize()
        {
            View = new PlotView
            {
                Dock = DockStyle.Fill,
            };

            Model = new PlotModel();

            heatMapSeries = new HeatMapSeries
            {
                X0 = 0,
                X1 = 100,
                Y0 = 0,
                Y1 = 100,
                Interpolate = true,
                RenderMethod = renderMethod,
                CoordinateDefinition = HeatMapCoordinateDefinition.Edge
            };

            colorAxis = new LinearColorAxis
            {
                Position = AxisPosition.Right,
            };

            Model.Axes.Add(colorAxis);
            Model.Series.Add(heatMapSeries);

            View.Model = Model;
            Controls.Add(View);

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

            VisualizerPropertiesDropDown = new ToolStripDropDownButton("Visualizer Properties");

            foreach (var item in toolStripItems)
            {
                VisualizerPropertiesDropDown.DropDownItems.Add(item);
            }

            statusStrip.Items.Add(VisualizerPropertiesDropDown);

            Controls.Add(statusStrip);
            View.MouseClick += (sender, e) =>
            {
                if (e.Button == MouseButtons.Right)
                {
                    statusStrip.Visible = !statusStrip.Visible;
                }
            };

            AutoScaleDimensions = new SizeF(6F, 13F);

            View.HandleDestroyed += (sender, e) =>
            {
                ValueMin = double.IsNaN(colorAxis.Minimum) ? null : colorAxis.Minimum;
                ValueMax = double.IsNaN(colorAxis.Maximum) ? null : colorAxis.Maximum;
            };
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
                Text = ValueMax.ToString(),
            };

            if (ValueMax.HasValue)
                colorAxis.Maximum = ValueMax.Value;

            maxValueTextBox.TextChanged += (sender, e) =>
            {

                if (double.TryParse(maxValueTextBox.Text, out double maxValue))
                {
                    colorAxis.Maximum = maxValue;
                }

                else if (string.IsNullOrEmpty(maxValueTextBox.Text))
                {
                    colorAxis.Maximum = double.NaN;
                }

                else
                {
                    maxValueTextBox.Text = colorAxis.Maximum.ToString();
                    return;
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
                Text = ValueMin.ToString()
            };

            if (ValueMin.HasValue)
                colorAxis.Minimum = ValueMin.Value;

            minValueTextBox.TextChanged += (sender, e) =>
            {
                if (double.TryParse(minValueTextBox.Text, out double minValue))
                {
                    colorAxis.Minimum = minValue;
                }

                else if (string.IsNullOrEmpty(minValueTextBox.Text))
                {
                    colorAxis.Minimum = double.NaN;
                }

                else
                {
                    minValueTextBox.Text = colorAxis.Minimum.ToString();
                    return;
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
            paletteComboBox.SelectedIndex = paletteSelectedIndex;
            UpdateColorPalette();
        }

        private void PaletteComboBoxSelectedIndexChanged(object sender, EventArgs e)
        {
            if (paletteComboBox.SelectedIndex != paletteSelectedIndex)
            {
                paletteSelectedIndex = paletteComboBox.SelectedIndex;
                UpdateColorPalette();
                PaletteComboBoxValueChanged?.Invoke(this, e);
                UpdatePlot();
            }
        }

        private void UpdateColorPalette()
        {
            var selectedPalette = (ColorPalette)paletteComboBox.Items[paletteSelectedIndex];
            paletteLookup.TryGetValue(selectedPalette, out Func<int, OxyPalette> paletteMethod);
            palette = paletteMethod(100);
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

            renderMethodComboBox.SelectedIndexChanged += RenderMethodComboBoxSelectedIndexChanged;
            renderMethodComboBox.SelectedIndex = renderMethodSelectedIndex;
            UpdateRenderMethod();
        }

        private void RenderMethodComboBoxSelectedIndexChanged(object sender, EventArgs e)
        {
            if (renderMethodComboBox.SelectedIndex != renderMethodSelectedIndex)
            {
                renderMethodSelectedIndex = renderMethodComboBox.SelectedIndex;
                UpdateRenderMethod();
                RenderMethodComboBoxValueChanged?.Invoke(this, e);
                UpdatePlot();
            }
        }

        private void UpdateRenderMethod()
        {
            renderMethod = (HeatMapRenderMethod)renderMethodComboBox.Items[renderMethodSelectedIndex];
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
            Model.InvalidatePlot(true);
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
