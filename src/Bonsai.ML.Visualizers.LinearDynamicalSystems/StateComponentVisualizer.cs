using System;
using System.Windows.Forms;
using System.Collections.Generic;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers.LinearDynamicalSystems;
using Bonsai.ML.LinearDynamicalSystems;
using OxyPlot;
using System.Reactive;
using OxyPlot.Series;
using System.Linq;

[assembly: TypeVisualizer(typeof(StateComponentVisualizer), Target = typeof(StateComponent))]

namespace Bonsai.ML.Visualizers.LinearDynamicalSystems
{
    /// <summary>
    /// Provides a type visualizer to display the state components of a Kalman Filter kinematics model.
    /// </summary>
    public class StateComponentVisualizer : BufferedVisualizer
    {
        internal TimeSeriesOxyPlotBase Plot;

        internal LineSeries LineSeries { get; private set; }

        internal AreaSeries AreaSeries { get; private set; }

        private bool resetAxes = true;

        private DateTime? startTime;

        /// <summary>
        /// Gets or sets the amount of time in seconds that should be shown along the x axis.
        /// </summary>
        public int Capacity { get; set; } = 10;

        /// <summary>
        /// Gets or sets a boolean value that determines whether to buffer the data beyond the capacity.
        /// </summary>
        public bool BufferData { get; set; } = true;

        /// <summary>
        /// Gets or sets the optional label to prepend to the line and area series names.
        /// </summary>
        public string Label { get; set; }

        /// <summary>
        /// Gets or sets the color to use for the line series.
        /// </summary>
        public OxyColor? LineSeriesColor { get; set; } = null;

        /// <summary>
        /// Gets or sets the color to use for the area series.
        /// </summary>
        public OxyColor? AreaSeriesColor { get; set; } = null;

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

            var lineSeriesName = string.IsNullOrEmpty(Label) ? "Mean" : $"{Label} Mean";
            LineSeries = Plot.AddNewLineSeries(lineSeriesName, color: LineSeriesColor);

            var areaSeriesName = string.IsNullOrEmpty(Label) ? "Variance" : $"{Label} Variance";
            AreaSeries = Plot.AddNewAreaSeries(areaSeriesName, color: AreaSeriesColor);

            if (Label != null) Plot.ValueLabel = Label;
            Plot.ResetLineSeries(LineSeries);
            Plot.ResetAreaSeries(AreaSeries);
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
            if (!startTime.HasValue)
            {
                startTime = time;
                Plot.StartTime = startTime.Value;
                Plot.ResetAxes();
            }

            StateComponent stateComponent = (StateComponent)value;
            double mean = stateComponent.Mean;
            double variance = stateComponent.Variance;

            Plot.AddToLineSeries(
                lineSeries: LineSeries,
                time: time,
                value: mean
            );

            Plot.AddToAreaSeries(
                areaSeries: AreaSeries,
                time: time,
                value1: mean + variance,
                value2: mean - variance
            );
        }

        /// <inheritdoc/>
        protected override void ShowBuffer(IList<Timestamped<object>> values)
        {
            base.ShowBuffer(values);
            if (values.Count > 0)
            {
                if (resetAxes)
                {
                    var time = values.LastOrDefault().Timestamp.DateTime;
                    Plot.SetAxes(minTime: time.AddSeconds(-Capacity), maxTime: time);
                }             
                Plot.UpdatePlot();
            }
        }

        internal void ShowDataBuffer(IList<Timestamped<object>> values, bool resetAxes = true)
        {
            this.resetAxes = resetAxes;
            ShowBuffer(values);
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            startTime = null;
            if (!Plot.IsDisposed)
            {
                Plot.Dispose();
            }
        }
    }
}
