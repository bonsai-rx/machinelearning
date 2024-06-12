using System;
using System.Windows.Forms;
using System.Collections.Generic;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers;
using Bonsai.ML.HiddenMarkovModels;
using OxyPlot;
using OxyPlot.Series;

[assembly: TypeVisualizer(typeof(StateObservationStatisticsVisualizer), Target = typeof(StateObservationStatistics))]

namespace Bonsai.ML.Visualizers
{
    public class StateObservationStatisticsVisualizer : DialogTypeVisualizer
    {
        
        private BarSeriesOxyPlotBase Plot;
        private List<ErrorBarSeries> allBarSeries = null;
        private StateObservationStatistics shown = null;
        
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
            if (value is StateObservationStatistics statistics && statistics != shown)
            {
                if (allBarSeries == null)
                {
                    allBarSeries = new List<ErrorBarSeries>();
                    var seriesCount = statistics.Means.GetLength(1);

                    for (int i = 0; i < statistics.Means.GetLength(1); i++)
                    {
                        OxyColor fillColor = OxyPalettes.Jet(seriesCount).Colors[i];
                        allBarSeries.Add(Plot.AddNewErrorBarSeries($"Dimension: {i}", fillColor: fillColor));
                    }
                }

                foreach (var barSeries in allBarSeries)
                {
                    Plot.ResetBarSeries(barSeries);
                }

                var minValue = 0.0;
                var maxValue = 0.0;
                var paddingPercentage = 0.05;

                for (int i = 0; i < statistics.Means.GetLength(0); i++)
                {
                    for (int j = 0; j < statistics.Means.GetLength(1); j++)
                    {
                        var val = statistics.Means[i, j];
                        var err = statistics.StdDevs[i, j];

                        minValue = Math.Min(minValue, val - err);
                        maxValue = Math.Max(maxValue, val + err);

                        Plot.AddValueAndErrorToBarSeries(allBarSeries[j], val, err);
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
