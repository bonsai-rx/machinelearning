using System.Windows.Forms;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.WindowsForms;
using System.Drawing;
using System;
using OxyPlot.Axes;
using System.Collections.Generic;

namespace Bonsai.ML.Visualizers
{
    internal class HeatMapSeriesOxyPlotBase : UserControl
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

        private int _numColors = 100;

        /// <summary>
        /// Event handler which can be used to hook into events generated when the combobox values have changed.
        /// </summary>
        public event EventHandler PaletteComboBoxValueChanged;

        public ToolStripComboBox PaletteComboBox
        {
            get => paletteComboBox;
        }

        public event EventHandler RenderMethodComboBoxValueChanged;

        public ToolStripComboBox RenderMethodComboBox
        {
            get => renderMethodComboBox;
        }

        public StatusStrip StatusStrip
        {
            get => statusStrip;
        }

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
            // palette = OxyPalettes.Rainbow(_numColors);
            Initialize();
        }

        private void Initialize()
        {
            view = new PlotView
            {
                Size = Size,
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

            InitilizeColorPalette();
            InitilizeRenderMethod();

            statusStrip = new StatusStrip
            {
                Visible = false
            };

            statusStrip.Items.AddRange(new ToolStripItem[] {
                paletteLabel,
                paletteComboBox,
                renderMethodLabel,
                renderMethodComboBox
            });

            Controls.Add(statusStrip);
            view.MouseClick += new MouseEventHandler(onMouseClick);
            AutoScaleDimensions = new SizeF(6F, 13F);
        }

        private void InitilizeColorPalette()
        {
            paletteLabel = new ToolStripLabel
            {
                Text = "Color palette:",
                AutoSize = true
            };

            paletteComboBox = new ToolStripComboBox()
            {
                Name = "palette",
                AutoSize = true,
            };

            foreach (var value in Enum.GetValues(typeof(ColorPalettes)))
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
            var selectedPalette = (ColorPalettes)paletteComboBox.Items[_paletteSelectedIndex];
            paletteLookup.TryGetValue(selectedPalette, out Func<int, OxyPalette> paletteMethod);
            palette = paletteMethod(_numColors);
            colorAxis.Palette = palette;
        }

        private void InitilizeRenderMethod()
        {
            renderMethodLabel = new ToolStripLabel
            {
                Text = "Render method:",
                AutoSize = true
            };

            renderMethodComboBox = new ToolStripComboBox()
            {
                Name = "renderMethod",
                AutoSize = true,
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

        public void UpdateHeatMapSeries(double x0, double x1, double y0, double y1, double[,] data)
        {
            heatMapSeries.X0 = x0;
            heatMapSeries.X1 = x1;
            heatMapSeries.Y0 = y0;
            heatMapSeries.Y1 = y1;
            heatMapSeries.Data = data;
        }

        public void UpdatePlot()
        {
            model.InvalidatePlot(true);
        }

        private static readonly Dictionary<ColorPalettes, Func<int, OxyPalette>> paletteLookup = new Dictionary<ColorPalettes, Func<int, OxyPalette>>
        {
            { ColorPalettes.Cividis, (numColors) => OxyPalettes.Cividis(numColors) },
            { ColorPalettes.Inferno, (numColors) => OxyPalettes.Inferno(numColors) },
            { ColorPalettes.Viridis, (numColors) => OxyPalettes.Viridis(numColors) },
            { ColorPalettes.Magma, (numColors) => OxyPalettes.Magma(numColors) },
            { ColorPalettes.Plasma, (numColors) => OxyPalettes.Plasma(numColors) },
            { ColorPalettes.BlackWhiteRed, (numColors) => OxyPalettes.BlackWhiteRed(numColors) },
            { ColorPalettes.BlueWhiteRed, (numColors) => OxyPalettes.BlueWhiteRed(numColors) },
            { ColorPalettes.Cool, (numColors) => OxyPalettes.Cool(numColors) },
            { ColorPalettes.Gray, (numColors) => OxyPalettes.Gray(numColors) },
            { ColorPalettes.Hot, (numColors) => OxyPalettes.Hot(numColors) },
            { ColorPalettes.Hue, (numColors) => OxyPalettes.Hue(numColors) },
            { ColorPalettes.HueDistinct, (numColors) => OxyPalettes.HueDistinct(numColors) },
            { ColorPalettes.Jet, (numColors) => OxyPalettes.Jet(numColors) },
            { ColorPalettes.Rainbow, (numColors) => OxyPalettes.Rainbow(numColors) },
        };
    }
}
