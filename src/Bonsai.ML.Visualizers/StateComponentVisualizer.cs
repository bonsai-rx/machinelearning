using System;
using System.Windows.Forms;
using System.Collections.Generic;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers;
using Bonsai.ML.LinearDynamicalSystems;
using System.Drawing;
using System.Reactive;
using OxyPlot.Series;

[assembly: TypeVisualizer(typeof(StateComponentVisualizer), Target = typeof(StateComponent))]

namespace Bonsai.ML.Visualizers
{
    /// <summary>
    /// Provides a type visualizer to display the state components of a Kalman Filter kinematics model.
    /// </summary>
    public class StateComponentVisualizer : BufferedVisualizer
    {

        private DateTime? _startTime;

        private TimeSeriesOxyPlotBase Plot;

        private LineSeries lineSeries;

        private AreaSeries areaSeries;

        /// <summary>
        /// Capacity or length of time shown along the x axis of the plot during automatic updating
        /// </summary>
        public int Capacity { get; set; } = 10;

        /// <summary>
        /// Buffer the data beyond the capacity.
        /// </summary>
        public bool BufferData { get; set; } = true;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            Plot = new TimeSeriesOxyPlotBase()
            {
                Capacity = Capacity,
                Dock = DockStyle.Fill,
                StartTime = DateTime.Now,
                BufferData = BufferData
            };

            lineSeries = Plot.AddNewLineSeries("Mean");
            areaSeries = Plot.AddNewAreaSeries("Variance");

            Plot.ResetLineSeries(lineSeries);
            Plot.ResetAreaSeries(areaSeries);
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
        protected override void Show(DateTime time, object value)
        {
            if (!_startTime.HasValue)
            {
                _startTime = time;
                Plot.StartTime = _startTime.Value;
                Plot.ResetAxes();
            }

            StateComponent stateComponent = (StateComponent)value;
            double mean = stateComponent.Mean;
            double variance = stateComponent.Variance;

            Plot.AddToLineSeries(
                lineSeries: lineSeries,
                time: time,
                value: mean
            );

            Plot.AddToAreaSeries(
                areaSeries: areaSeries,
                time: time,
                value1: mean + variance,
                value2: mean - variance
            );

            Plot.SetAxes(minTime: time.AddSeconds(-Capacity), maxTime: time);

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
