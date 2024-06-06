using System;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Reflection;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers;
using Bonsai.ML.HiddenMarkovModels;
using Bonsai.ML.LinearDynamicalSystems.LinearRegression;
using System.Drawing;
using System.Reactive;
using Bonsai.Reactive;
using OxyPlot;
using OxyPlot.Series;
using System.Linq;

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

                        minValue = Math.Min(minValue, val);
                        maxValue = Math.Max(maxValue, val);

                        Plot.AddValueAndErrorToBarSeries(allBarSeries[j], statistics.Means[i, j], statistics.StdDevs[i, j]);
                    }
                }

                Plot.SetAxes(minValue - (minValue * paddingPercentage), maxValue + (maxValue * paddingPercentage));

                shown = statistics;
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
