using Bonsai.Design;
using Bonsai;
using Bonsai.ML.Visualizers;
using Bonsai.ML.LinearDynamicalSystems;
using Bonsai.ML.LinearDynamicalSystems.Kinematics;
using System;
using System.Collections.Generic;
using OxyPlot.Series;
using OxyPlot;

[assembly: TypeVisualizer(typeof(ForecastPlotOverlay), Target = typeof(MashupSource<KinematicStateVisualizer, ForecastVisualizer>))]

namespace Bonsai.ML.Visualizers
{
    /// <summary>
    /// Provides a mashup visualizer to display the forecast of a Kalman Filter kinematics model overtime of a KinematicStateVisualizer.
    /// </summary>
    public class ForecastPlotOverlay : DialogTypeVisualizer
    {
        private TimeSeriesOxyPlotBase plot;

        private LineSeries lineSeries;

        private AreaSeries areaSeries;

        private KinematicStateVisualizer visualizer;

        /// <inheritdoc/>
        public override void Show(object value)
        {
            var time = DateTime.Now;
            plot.ResetLineSeries(lineSeries);
            plot.ResetAreaSeries(areaSeries);

            Forecast forecast = (Forecast)value;
            List<ForecastResult> forecastResults = forecast.ForecastResults;
            DateTime forecastTime = time;

            for (int i = 0; i < forecastResults.Count; i++)
            {
                var forecastResult = forecastResults[i];
                var kinematicState = forecastResult.KinematicState;

                forecastTime = time + forecastResult.Timestep;

                KinematicComponent kinematicComponent = (KinematicComponent)visualizer.kinematicComponentProperty.GetValue(kinematicState);
                StateComponent stateComponent = (StateComponent)visualizer.stateComponentProperty.GetValue(kinematicComponent);

                double mean = stateComponent.Mean;
                double variance = stateComponent.Variance;

                plot.AddToLineSeries(
                    lineSeries: lineSeries,
                    time: forecastTime,
                    value: mean
                );

                plot.AddToAreaSeries(
                    areaSeries: areaSeries,
                    time: forecastTime,
                    value1: mean + variance,
                    value2: mean - variance
                );
            }

            plot.SetAxes(minTime: forecastTime.AddSeconds(-plot.Capacity), maxTime: forecastTime);
        }
        
        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            var service = provider.GetService(typeof(MashupVisualizer));
            visualizer = (KinematicStateVisualizer)service;
            plot = visualizer.Plot;

            lineSeries = plot.AddNewLineSeries("Forecast Mean", color: OxyColors.Yellow);
            areaSeries = plot.AddNewAreaSeries("Forecast Variance", color: OxyColors.Yellow, opacity: 50);

            plot.ResetLineSeries(lineSeries);
            plot.ResetAreaSeries(areaSeries);
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            plot.ResetLineSeries(lineSeries);
            plot.ResetAreaSeries(areaSeries);
        }

    }
}
