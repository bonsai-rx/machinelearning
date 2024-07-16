using System;
using System.Windows.Forms;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers.LinearDynamicalSystems;
using Bonsai.ML.LinearDynamicalSystems.LinearRegression;

[assembly: TypeVisualizer(typeof(MultivariatePDFVisualizer), Target = typeof(MultivariatePDF))]

namespace Bonsai.ML.Visualizers.LinearDynamicalSystems
{

    /// <summary>
    /// Provides a type visualizer to display a multivariate probability distribution as a heatmap.
    /// </summary>
    public class MultivariatePDFVisualizer : DialogTypeVisualizer
    {
        /// <summary>
        /// Gets or sets the selected index of the color palette to use.
        /// </summary>
        public int PaletteSelectedIndex { get; set; }

        /// <summary>
        /// Gets or sets the selected index of the render method to use.
        /// </summary>
        public int RenderMethodSelectedIndex { get; set; }

        private HeatMapSeriesOxyPlotBase Plot;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            Plot = new HeatMapSeriesOxyPlotBase(PaletteSelectedIndex, RenderMethodSelectedIndex)
            {
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
            var pdf = (MultivariatePDF)value;
            Plot.UpdateHeatMapSeries(
                pdf.GridParameters.X0 - (1 / 2 * pdf.GridParameters.XSteps),
                pdf.GridParameters.X1 - (1 / 2 * pdf.GridParameters.XSteps),
                pdf.GridParameters.Y0 - (1 / 2 * pdf.GridParameters.YSteps),
                pdf.GridParameters.Y1 - (1 / 2 * pdf.GridParameters.YSteps),
                pdf.Values
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
