using System;
using System.Windows.Forms;
using System.Collections.Generic;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers;
using Bonsai.ML.LinearDynamicalSystems.Kinematics;
using OxyPlot;
using System.Reactive;
using System.Linq;
using System.Reactive.Linq;

[assembly: TypeVisualizer(typeof(ForecastVisualizer), Target = typeof(Forecast))]

namespace Bonsai.ML.Visualizers
{
    /// <summary>
    /// Provides a type visualizer to display the forecast of a Kalman Filter kinematics model.
    /// </summary>
    public class ForecastVisualizer : BufferedVisualizer
    {

        internal int RowCount { get; set; } = 3;
        internal int ColumnCount { get; set; } = 2;
        internal string[] Labels = new string[] { 
            "Forecast Position X", 
            "Forecast Position Y", 
            "Forecast Velocity X", 
            "Forecast Velocity Y", 
            "Forecast Acceleration X", 
            "Forecast Acceleration Y" 
        };

        internal List<StateComponentVisualizer> componentVisualizers = new();
        private TableLayoutPanel container;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            container = new TableLayoutPanel
            {
                ColumnCount = ColumnCount,
                RowCount = RowCount,
                Dock = DockStyle.Fill
            };

            for (int i = 0; i < container.RowCount; i++)
            {
                container.RowStyles.Add(new RowStyle(SizeType.Percent, 100f / RowCount));
            }

            for (int i = 0; i < container.ColumnCount; i++)
            {
                container.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f / ColumnCount));
            }

            for (int i = 0 ; i < RowCount; i++)
            {
                for (int j = 0; j < ColumnCount; j++)
                {
                    var StateComponentVisualizer = new StateComponentVisualizer() {
                        Label = Labels[i * ColumnCount + j],
                        LineSeriesColor = OxyColors.Yellow,
                        AreaSeriesColor = OxyColors.Yellow
                    };
                    StateComponentVisualizer.Load(provider);
                    container.Controls.Add(StateComponentVisualizer.Plot, j, i);
                    componentVisualizers.Add(StateComponentVisualizer);
                }
            }

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));

            if (visualizerService != null)
            {
                visualizerService.AddControl(container);
            }
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
        }

        /// <inheritdoc/>
        protected override void ShowBuffer(IList<Timestamped<object>> values)
        {
            if (values.Count == 0) return;
            var latestForecast = values.Last();
            var timestamp = latestForecast.Timestamp;
            var forecast = (Forecast)latestForecast.Value;
            var futureTime = timestamp;

            List<Timestamped<object>> positionX = new();
            List<Timestamped<object>> positionY = new();
            List<Timestamped<object>> velocityX = new();
            List<Timestamped<object>> velocityY = new();
            List<Timestamped<object>> accelerationX = new();
            List<Timestamped<object>> accelerationY = new();

            foreach (var forecastResult in forecast.ForecastResults)
            {
                futureTime = timestamp + forecastResult.Timestep;
                positionX.Add(new Timestamped<object>(forecastResult.KinematicState.Position.X, futureTime));
                positionY.Add(new Timestamped<object>(forecastResult.KinematicState.Position.Y, futureTime));
                velocityX.Add(new Timestamped<object>(forecastResult.KinematicState.Velocity.X, futureTime));
                velocityY.Add(new Timestamped<object>(forecastResult.KinematicState.Velocity.Y, futureTime));
                accelerationX.Add(new Timestamped<object>(forecastResult.KinematicState.Acceleration.X, futureTime));
                accelerationY.Add(new Timestamped<object>(forecastResult.KinematicState.Acceleration.Y, futureTime));
            }

            var dataList = new List<List<Timestamped<object>>>() { positionX, positionY, velocityX, velocityY, accelerationX, accelerationY };

            var zippedData = dataList.Zip(componentVisualizers, (data, visualizer) => new { Data = data, Visualizer = visualizer });

            foreach (var item in zippedData)
            {
                item.Visualizer.Plot.ResetLineSeries(item.Visualizer.lineSeries);
                item.Visualizer.Plot.ResetAreaSeries(item.Visualizer.areaSeries);
                item.Visualizer.ShowDataBuffer(item.Data);
                item.Visualizer.Plot.SetAxes(minTime: timestamp.DateTime, maxTime: futureTime.DateTime);
            }
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            foreach (var componentVisualizer in componentVisualizers) componentVisualizer.Unload();
            if (componentVisualizers.Count > 0) componentVisualizers.Clear();
            if (!container.IsDisposed) container.Dispose();
        }
    }
}