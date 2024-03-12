using System.Windows.Forms;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.WindowsForms;
using System.Drawing;
using System;
using OxyPlot.Axes;
using System.Collections.Generic;
using System.Reflection;

namespace Bonsai.ML.Visualizers
{
    internal class HeatMapSeriesOxyPlotBase : UserControl
    {
        private PlotView view;
        private PlotModel model;
        private HeatMapSeries heatMapSeries;
        private LinearColorAxis colorAxis;

        private ComboBox comboBox;
        private Label label;

        private int _numColors = 100;

        private int _selectedIndex;

        private OxyPalette palette;

        /// <summary>
        /// Event handler which can be used to hook into events generated when the combobox values have changed.
        /// </summary>
        public event EventHandler ComboBoxValueChanged;

        public ComboBox ComboBox
        {
            get => comboBox;
        }

        /// <summary>
        /// Constructor of the TimeSeriesOxyPlotBase class.
        /// Requires a line series name and an area series name.
        /// Data source is optional, since pasing it to the constructor will populate the combobox and leave it empty otherwise.
        /// The selected index is only needed when the data source is provided.
        /// </summary>
        public HeatMapSeriesOxyPlotBase(int selectedIndex, int numColors = 100)
        {
            _selectedIndex = selectedIndex;
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
                RenderMethod = HeatMapRenderMethod.Bitmap,
                CoordinateDefinition = HeatMapCoordinateDefinition.Center
            };

            colorAxis = new LinearColorAxis {
                Position = AxisPosition.Right,
            };

            model.Axes.Add(colorAxis);
            model.Series.Add(heatMapSeries);

            view.Model = model;
            Controls.Add(view);

            label = new Label
            {
                Text = "Color palette:",
                AutoSize = true,
                Location = new Point(-200, 8),
                Anchor = AnchorStyles.Top | AnchorStyles.Right
            };

            comboBox = new ComboBox
            {
                Location = new Point(0, 5),
                DataSource = Enum.GetValues(typeof(ColorPalettes)),
                Anchor = AnchorStyles.Top | AnchorStyles.Right,
                BindingContext = BindingContext
            };

            Controls.Add(comboBox);
            Controls.Add(label);

            comboBox.SelectedIndexChanged += ComboBoxSelectedIndexChanged;
            comboBox.SelectedIndex = _selectedIndex;
            UpdateColorPalette();

            comboBox.BringToFront();
            label.BringToFront();

            AutoScaleDimensions = new SizeF(6F, 13F);
        }

        private void ComboBoxSelectedIndexChanged(object sender, EventArgs e)
        {
            if (comboBox.SelectedIndex != _selectedIndex)
            {
                _selectedIndex = comboBox.SelectedIndex;
                UpdateColorPalette();
                ComboBoxValueChanged?.Invoke(this, e);
                UpdatePlot();
            }
        }

        private void UpdateColorPalette()
        {
            var selectedPalette = (ColorPalettes)comboBox.Items[_selectedIndex];
            paletteLookup.TryGetValue(selectedPalette, out Func<int, OxyPalette> paletteMethod);
            palette = paletteMethod(_numColors);
            colorAxis.Palette = palette;
        }

        public void UpdateHeatMapSeries(double x0, double x1, int xsteps, double y0, double y1, int ysteps, double[,] data)
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

    internal enum ColorPalettes
    {
        Cividis,
        Inferno,
        Viridis,
        Magma,
        Plasma,
        BlackWhiteRed,
        BlueWhiteRed,
        Cool,
        Gray,
        Hot,
        Hue,
        HueDistinct,
        Jet,
        Rainbow
    }
}
