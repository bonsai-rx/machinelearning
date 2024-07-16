using System;
using System.Windows.Forms;
using System.Collections.Generic;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers.HiddenMarkovModels;
using Bonsai.ML.HiddenMarkovModels.Observations;
using OxyPlot;
using OxyPlot.Series;

[assembly: TypeVisualizer(typeof(GaussianObservationStatisticsVisualizer), Target = typeof(GaussianObservationsStatistics))]

namespace Bonsai.ML.Visualizers.HiddenMarkovModels
{
    /// <summary>
    /// Provides a type visualizer of <see cref="GaussianObservationsStatistics"/> to display the means and standard 
    /// deviations of each state of an HMM with gaussian observations model.
    /// </summary>
    public class GaussianObservationStatisticsVisualizer : DialogTypeVisualizer
    {
        
        private BarSeriesOxyPlotBase Plot;
        private List<ErrorBarSeries> allBarSeries = null;
        private GaussianObservationsStatistics shown = null;
        
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

            if (shown != null)
            {
                var value = shown;
                shown = null;
                Show(value);
            }
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            if (value is GaussianObservationsStatistics statistics && statistics != shown)
            {
                if (statistics.Means == null || statistics.StdDevs == null)
                {
                    return;
                }
                
                if (allBarSeries == null)
                {
                    allBarSeries = new List<ErrorBarSeries>();
                    var seriesCount = statistics.Means.GetLength(1);

                    for (int i = 0; i < seriesCount; i++)
                    {
                        allBarSeries.Add(Plot.AddNewErrorBarSeries($"Dimension: {i}", strokeColor: OxyColors.Black));
                    }
                }

                foreach (var barSeries in allBarSeries)
                {
                    Plot.ResetBarSeries(barSeries);
                }

                var minValue = 0.0;
                var maxValue = 0.0;
                var paddingPercentage = 0.05;

                var nStates = statistics.Means.GetLength(0);
                var nDims = statistics.Means.GetLength(1);

                for (int i = 0; i < nStates; i++)
                {
                    OxyColor fillColor = OxyPalettes.Jet(nStates).Colors[i];
                    for (int j = 0; j < nDims; j++)
                    {
                        var val = statistics.Means[i, j];
                        var err = statistics.StdDevs[i, j];

                        minValue = Math.Min(minValue, val - err);
                        maxValue = Math.Max(maxValue, val + err);

                        Plot.AddValueAndErrorToBarSeries(allBarSeries[j], val, err, fillColor: fillColor);
                    }
                }

                var pad = Math.Max(Math.Abs(minValue), Math.Abs(maxValue)) * paddingPercentage;

                Plot.SetAxes(minValue - pad, maxValue + pad);

                shown = statistics;

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
