using System;
using System.Windows.Forms;
using Bonsai;
using Bonsai.Design;

[assembly: TypeVisualizer(typeof(Bonsai.ML.Design.MultidimensionalArrayVisualizer),
    Target = typeof(double[,]))]

namespace Bonsai.ML.Design
{
    /// <summary>
    /// Provides a type visualizer to display multi dimensional array data as a heatmap.
    /// </summary>
    public class MultidimensionalArrayVisualizer : DialogTypeVisualizer
    {
        /// <summary>
        /// Gets or sets the selected index of the color palette to use.
        /// </summary>
        public int PaletteSelectedIndex { get; set; }

        /// <summary>
        /// Gets or sets the selected index of the render method to use.
        /// </summary>
        public int RenderMethodSelectedIndex { get; set; }

        private HeatMapSeriesOxyPlotBase _plot;
        /// <summary>
        /// Gets the HeatMapSeriesOxyPlotBase control used to display the heatmap.
        /// </summary>
        public HeatMapSeriesOxyPlotBase Plot => _plot;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            _plot = new HeatMapSeriesOxyPlotBase(PaletteSelectedIndex, RenderMethodSelectedIndex)
            {
                Dock = DockStyle.Fill,
            };

            _plot.PaletteComboBoxValueChanged += PaletteIndexChanged;
            _plot.RenderMethodComboBoxValueChanged += RenderMethodIndexChanged;

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            visualizerService?.AddControl(_plot);
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

        private void PaletteIndexChanged(object sender, EventArgs e)
        {
            PaletteSelectedIndex = Plot.PaletteComboBox.SelectedIndex;
        }
        
        private void RenderMethodIndexChanged(object sender, EventArgs e)
        {
            RenderMethodSelectedIndex = Plot.RenderMethodComboBox.SelectedIndex;
        }
    }
}
