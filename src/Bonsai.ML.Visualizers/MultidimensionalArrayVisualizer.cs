using System;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Reflection;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers;
using Bonsai.ML.LinearDynamicalSystems;
using Bonsai.ML.LinearDynamicalSystems.LinearRegression;
using System.Drawing;
using System.Reactive;
using Bonsai.Reactive;
using Bonsai.Expressions;
using OxyPlot;

[assembly: TypeVisualizer(typeof(MultidimensionalArrayVisualizer), Target = typeof(double[,]))]

namespace Bonsai.ML.Visualizers
{

    /// <summary>
    /// Provides a type visualizer to display the state components of a Kalman Filter kinematics model.
    /// </summary>
    public class MultidimensionalArrayVisualizer : DialogTypeVisualizer
    {

        private int paletteSelectedIndex = 0;
        private int renderMethodSelectedIndex = 0;

        /// <summary>
        /// Size of the window when loaded
        /// </summary>
        public Size Size { get; set; } = new Size(320, 240);

        /// <summary>
        /// The selected index of the color palette to use
        /// </summary>
        public int PaletteSelectedIndex { get => paletteSelectedIndex; set => paletteSelectedIndex = value; }
        public int RenderMethodSelectedIndex { get => renderMethodSelectedIndex; set => renderMethodSelectedIndex = value; }

        private HeatMapSeriesOxyPlotBase Plot;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            Plot = new HeatMapSeriesOxyPlotBase(paletteSelectedIndex, renderMethodSelectedIndex)
            {
                Size = Size,
                Dock = DockStyle.Fill,
            };

            Plot.PaletteComboBoxValueChanged += PaletteIndexChanged;
            Plot.RenderMethodComboBoxValueChanged += RenderMethodIndexChanged;

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            if (visualizerService != null)
            {
                visualizerService.AddControl(Plot);
            }
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            var mdarray = (double[,])value;
            var shape = new int[] {mdarray.GetLength(0), mdarray.GetLength(1)};

            Plot.UpdateHeatMapSeries(
                -0.5,
                shape[0] - 0.5,
                -0.5,
                shape[1] - 0.5,
                mdarray
            );

            Plot.UpdatePlot();
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            if (!Plot.IsDisposed)
            {
                Plot.Dispose();
            }
        }

        /// <summary>
        /// Callback function to update the selected index when the selected combobox index has changed
        /// </summary>
        private void PaletteIndexChanged(object sender, EventArgs e)
        {
            var comboBox = Plot.PaletteComboBox;
            paletteSelectedIndex = comboBox.SelectedIndex;
        }
        private void RenderMethodIndexChanged(object sender, EventArgs e)
        {
            var comboBox = Plot.RenderMethodComboBox;
            renderMethodSelectedIndex = comboBox.SelectedIndex;
        }
    }
}
