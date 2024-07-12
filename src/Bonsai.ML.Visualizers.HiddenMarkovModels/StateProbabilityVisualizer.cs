using System;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Linq;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers;
using Bonsai.ML.HiddenMarkovModels;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;

[assembly: TypeVisualizer(typeof(StateProbabilityVisualizer), Target = typeof(StateProbability))]

namespace Bonsai.ML.Visualizers
{
    /// <summary>
    /// Provides a type visualizer of <see cref="StateProbability"/> to display the probabilities 
    /// of being in each state of an HMM given the observation.
    /// </summary>
    public class StateProbabilityVisualizer : DialogTypeVisualizer
    {
        private BarSeriesOxyPlotBase Plot;

        private List<BarSeries> allBarSeries;
        
        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            Plot = new BarSeriesOxyPlotBase()
            {
                Dock = DockStyle.Fill,
            };

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));

            if (visualizerService != null)
            {
                visualizerService.AddControl(Plot);
            }
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            if (value is StateProbability stateProbability)
            {
                if (allBarSeries == null)
                {
                    allBarSeries = new List<BarSeries>();
                    var statesCount = stateProbability.Probabilities.Length;

                    for (int i = 0; i < statesCount; i++)
                    {
                        OxyColor fillColor = OxyPalettes.Jet(statesCount).Colors[i];
                        allBarSeries.Add(Plot.AddNewBarSeries($"State: {i}", fillColor: fillColor, strokeColor: OxyColors.Black));
                    }
                }

                foreach (var barSeries in allBarSeries)
                {
                    Plot.ResetBarSeries(barSeries);
                }

                var minValue = 0.0;
                var maxValue = 0.0;
                var paddingPercentage = 0.05;

                var nStates = stateProbability.Probabilities.Length;
                CategoryAxis categoryAxis = (CategoryAxis)Plot.XAxis;
                categoryAxis.ItemsSource = Enumerable.Range(0, nStates);

                for (int i = 0; i < nStates; i++)
                {
                    var val = stateProbability.Probabilities[i];

                    minValue = Math.Min(minValue, val);
                    maxValue = Math.Max(maxValue, val);

                    Plot.AddValueToBarSeries(allBarSeries[i], i, val);
                }

                var pad = Math.Max(Math.Abs(minValue), Math.Abs(maxValue)) * paddingPercentage;

                Plot.SetAxes(minValue - pad, maxValue + pad);

                Plot.UpdatePlot();
            }
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            allBarSeries = null;
            if (!Plot.IsDisposed)
            {
                Plot.Dispose();
            }
        }
    }
}
