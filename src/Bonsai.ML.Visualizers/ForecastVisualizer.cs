using System;
using System.Windows.Forms;
using System.Collections.Generic;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers;
using Bonsai.ML.LinearDynamicalSystems.Kinematics;
using System.Drawing;
using OxyPlot;
using OxyPlot.Series;
using System.Reactive;

[assembly: TypeVisualizer(typeof(ForecastVisualizer), Target = typeof(Forecast))]

namespace Bonsai.ML.Visualizers
{
    /// <summary>
    /// Provides a type visualizer to display the forecast of a Kalman Filter kinematics model.
    /// </summary>
    public class ForecastVisualizer : BufferedVisualizer
    {

        private DateTime? _startTime;
        private DateTime? lastUpdate = null;

        private TimeSeriesOxyPlotBase Plot;
        private LineSeries lineSeries;
        private AreaSeries areaSeries;

        /// <summary>
        /// Capacity or length of time shown along the x axis of the plot during automatic updating
        /// </summary>
        public int Capacity { get; set; } = 10;

        /// <summary>
        /// Buffer the data beyond the capacity
        /// </summary>
        public bool BufferData { get; set; } = true;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            Plot = new TimeSeriesOxyPlotBase()
            {
                Size = Size,
                Capacity = Capacity,
                Dock = DockStyle.Fill,
                StartTime = DateTime.Now,
                BufferData = BufferData,
            };

            lineSeries = Plot.AddNewLineSeries("Forecast Mean", color: OxyColors.Yellow);
            areaSeries = Plot.AddNewAreaSeries("Forecast Variance", color: OxyColors.Yellow);

            Plot.ResetAxes();

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            if (visualizerService != null)
            {
                visualizerService.AddControl(Plot);
            }
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
        }

        /// <inheritdoc/>
        public override void Show(DateTime time, object value)
        {
            if (!_startTime.HasValue)
            {
                _startTime = time;
                Plot.StartTime = _startTime.Value;
                Plot.ResetAxes();
            }

            Plot.ResetLineSeries(lineSeries);
            Plot.ResetAreaSeries(areaSeries);

            Forecast forecast = (Forecast)value;
            List<ForecastResult> forecastResults = forecast.ForecastResults;
            DateTime forecastTime = time;

            for (int i = 0; i < forecastResults.Count; i++)
            {
                var forecastResult = forecastResults[i];
                var kinematicState = forecastResult.KinematicState;

                forecastTime = time + forecastResult.Timestep;
                double mean = kinematicState.Position.X.Mean;
                double variance = kinematicState.Position.X.Variance;

                Plot.AddToLineSeries(
                    lineSeries: lineSeries,
                    time: forecastTime,
                    value: mean
                );

                Plot.AddToAreaSeries(
                    areaSeries: areaSeries,
                    time: forecastTime,
                    value1: mean + variance,
                    value2: mean - variance
                );
            }

            Plot.SetAxes(minTime: forecastTime.AddSeconds(-Capacity), maxTime: forecastTime);
        }

        /// <inheritdoc/>
        protected override void ShowBuffer(IList<Timestamped<object>> values)
        {
            base.ShowBuffer(values);
            if (values.Count > 0)
            {
                Plot.UpdatePlot();
            }
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            _startTime = null;
            if (!Plot.IsDisposed)
            {
                Plot.Dispose();
            }
        }
    }
}