using Bonsai.Design;
using Bonsai;
using Bonsai.ML.Visualizers.LinearDynamicalSystems;
using Bonsai.ML.LinearDynamicalSystems;
using Bonsai.ML.LinearDynamicalSystems.Kinematics;
using System;
using System.Collections.Generic;
using OxyPlot.Series;
using OxyPlot;

[assembly: TypeVisualizer(typeof(ForecastPlotOverlay), Target = typeof(MashupSource<KinematicStateVisualizer, ForecastVisualizer>))]

namespace Bonsai.ML.Visualizers.LinearDynamicalSystems
{
    /// <summary>
    /// Provides a mashup visualizer to display the forecast of a Kalman Filter kinematics model alongside the kinematic state.
    /// </summary>
    public class ForecastPlotOverlay : DialogTypeVisualizer
    {
        private List<LineSeries> lineSeriesList = new();

        private List<AreaSeries> areaSeriesList = new();

        private KinematicStateVisualizer visualizer;

        /// <inheritdoc/>
        public override void Show(object value)
        {
            var time = DateTime.Now;
            Forecast forecast = (Forecast)value;
            var componentVisualizers = visualizer.ComponentVisualizers;

            for (int i = 0; i < componentVisualizers.Count; i++)
            {
                var plot = componentVisualizers[i].Plot;
                var lineSeries = lineSeriesList[i];
                var areaSeries = areaSeriesList[i];

                plot.ResetLineSeries(lineSeries);
                plot.ResetAreaSeries(areaSeries);

                DateTime forecastTime = time;

                for (int j = 0; j < forecast.ForecastResults.Count; j++)
                {
                    var forecastResult = forecast.ForecastResults[j];
                    var kinematicState = forecastResult.KinematicState;
                    forecastTime = time + forecastResult.Timestep;

                    StateComponent[] stateComponents = new StateComponent[] {kinematicState.Position.X, kinematicState.Position.Y, kinematicState.Velocity.X, kinematicState.Velocity.Y, kinematicState.Acceleration.X, kinematicState.Acceleration.Y};

                    AddStateComponentDataToSeries(plot, stateComponents[i], lineSeries, areaSeries, forecastTime);                    

                }

                plot.SetAxes(minTime: forecastTime.AddSeconds(-plot.Capacity), maxTime: forecastTime);
            }
        }

        private void AddStateComponentDataToSeries(TimeSeriesOxyPlotBase plot, StateComponent stateComponent, LineSeries lineSeries, AreaSeries areaSeries, DateTime time)
        {
            double mean = stateComponent.Mean;
            double variance = stateComponent.Variance;

            plot.AddToLineSeries(
                lineSeries: lineSeries,
                time: time,
                value: mean
            );

            plot.AddToAreaSeries(
                areaSeries: areaSeries,
                time: time,
                value1: mean + variance,
                value2: mean - variance
            );
        }
        
        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            if (lineSeriesList.Count > 0)
            {
                lineSeriesList.Clear();
                lineSeriesList = new();
            }

            if (areaSeriesList.Count > 0)
            {
                areaSeriesList.Clear();
                areaSeriesList = new();
            }

            var service = provider.GetService(typeof(MashupVisualizer));
            visualizer = (KinematicStateVisualizer)service;
            var componentVisualizers = visualizer.ComponentVisualizers;

            for (int i = 0; i < componentVisualizers.Count; i++)
            {
                var lineSeries = componentVisualizers[i].Plot.AddNewLineSeries($"Forecast {visualizer.Labels[i]} Mean", color: OxyColors.Yellow);
                var areaSeries = componentVisualizers[i].Plot.AddNewAreaSeries($"Forecast {visualizer.Labels[i]} Variance", color: OxyColors.Yellow, opacity: 50);

                componentVisualizers[i].Plot.ResetLineSeries(lineSeries);
                componentVisualizers[i].Plot.ResetAreaSeries(areaSeries);

                lineSeriesList.Add(lineSeries);
                areaSeriesList.Add(areaSeries);
            }
        }

        /// <inheritdoc/>
        public override void Unload()
        {
        }
    }
}
