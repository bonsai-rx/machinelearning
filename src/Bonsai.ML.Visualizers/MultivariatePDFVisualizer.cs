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

[assembly: TypeVisualizer(typeof(MultivariatePDFVisualizer), Target = typeof(MultivariatePDF))]

namespace Bonsai.ML.Visualizers
{

    /// <summary>
    /// Provides a type visualizer to display the state components of a Kalman Filter kinematics model.
    /// </summary>
    public class MultivariatePDFVisualizer : DialogTypeVisualizer
    {

        private int selectedIndex = 0;

        /// <summary>
        /// Size of the window when loaded
        /// </summary>
        public Size Size { get; set; } = new Size(320, 240);

        /// <summary>
        /// The selected index of the color palette to use
        /// </summary>
        public int SelectedIndex { get => selectedIndex; set => selectedIndex = value; }

        private HeatMapSeriesOxyPlotBase Plot;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            Plot = new HeatMapSeriesOxyPlotBase(selectedIndex)
            {
                Size = Size,
                Dock = DockStyle.Fill,
            };

            Plot.ComboBoxValueChanged += IndexChanged;

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
                pdf.GridParameters.X0,
                pdf.GridParameters.X1,
                pdf.GridParameters.Xsteps,
                pdf.GridParameters.Y0,
                pdf.GridParameters.Y1,
                pdf.GridParameters.Ysteps,
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

        /// <summary>
        /// Callback function to update the selected index when the selected combobox index has changed
        /// </summary>
        private void IndexChanged(object sender, EventArgs e)
        {
            var comboBox = Plot.ComboBox;
            selectedIndex = comboBox.SelectedIndex;
        }
    }
}
