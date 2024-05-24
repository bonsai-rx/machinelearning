using Bonsai.Design;
using Bonsai;
using Bonsai.ML.LinearDynamicalSystems;
using Bonsai.ML.LinearDynamicalSystems.Kinematics;
using Bonsai.ML.Visualizers;
using System.Collections.Generic;
using System;
using System.Windows.Forms;
using System.Xml.Serialization;
using System.Reactive.Linq;
using OxyPlot.Series;
using System.Linq;
using System.Drawing;
using System.Reflection;
using System.Reactive;

[assembly: TypeVisualizer(typeof(KinematicStateVisualizer), Target = typeof(KinematicState))]

namespace Bonsai.ML.Visualizers
{

    /// <summary>
    /// Provides a type visualizer to display the state components of a Kalman Filter Kinematics model.
    /// </summary>
    public class KinematicStateVisualizer : MashupVisualizer
    {
        internal int RowCount { get; set; } = 3;
        internal int ColumnCount { get; set; } = 2;
        internal List<StateComponentVisualizer> componentVisualizers = new();
        private TableLayoutPanel container;
        private int updateFrequency = 1000 / 50;
        private bool resetAxes = true;

        internal string[] Labels = new string[] { 
            "Position X", 
            "Position Y", 
            "Velocity X", 
            "Velocity Y", 
            "Acceleration X", 
            "Acceleration Y" 
        };

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
                        Label = Labels[i * ColumnCount + j]
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

            base.Load(provider);
        }


        /// <inheritdoc/>
        public override void Show(object value)
        {
        }

        /// <inheritdoc/>
        public void ShowBuffer(IList<Timestamped<object>> values)
        {
            List<Timestamped<object>> positionX = new();
            List<Timestamped<object>> positionY = new();
            List<Timestamped<object>> velocityX = new();
            List<Timestamped<object>> velocityY = new();
            List<Timestamped<object>> accelerationX = new();
            List<Timestamped<object>> accelerationY = new();

            foreach (var value in values)
            {
                positionX.Add(new Timestamped<object>(((KinematicState)value.Value).Position.X, value.Timestamp));
                positionY.Add(new Timestamped<object>(((KinematicState)value.Value).Position.Y, value.Timestamp));
                velocityX.Add(new Timestamped<object>(((KinematicState)value.Value).Velocity.X, value.Timestamp));
                velocityY.Add(new Timestamped<object>(((KinematicState)value.Value).Velocity.Y, value.Timestamp));
                accelerationX.Add(new Timestamped<object>(((KinematicState)value.Value).Acceleration.X, value.Timestamp));
                accelerationY.Add(new Timestamped<object>(((KinematicState)value.Value).Acceleration.Y, value.Timestamp));
            }

            componentVisualizers[0].ShowDataBuffer(positionX, resetAxes);
            componentVisualizers[1].ShowDataBuffer(positionY, resetAxes);
            componentVisualizers[2].ShowDataBuffer(velocityX, resetAxes);
            componentVisualizers[3].ShowDataBuffer(velocityY, resetAxes);
            componentVisualizers[4].ShowDataBuffer(accelerationX, resetAxes);
            componentVisualizers[5].ShowDataBuffer(accelerationY, resetAxes);
        }

        /// <inheritdoc/>
        public override IObservable<object> Visualize(IObservable<IObservable<object>> source, IServiceProvider provider)
        {
            if (provider.GetService(typeof(IDialogTypeVisualizerService)) is not Control visualizerControl)
            {
                return source;
            }

            var visualizerSource = VisualizeSource(source, visualizerControl);

            if (MashupSources.Count == 0)
            {
                resetAxes = true;
                return visualizerSource;
            }

            resetAxes = false;

            var mashupSourceStreams = MashupSources.Select(mashupSource =>
                mashupSource.Visualizer.Visualize(mashupSource.Source.Output, provider)
                    .Do(value => mashupSource.Visualizer.Show(value)));

            var mergedMashupSources = Observable.Merge(mashupSourceStreams);

            return Observable.Merge(mergedMashupSources, visualizerSource);
        }

        private IObservable<object> VisualizeSource(IObservable<IObservable<object>> source, Control visualizerControl)
        {
            return Observable.Using(
                () => new Timer(),
                timer =>
                {
                    timer.Interval = updateFrequency;
                    var timerTick = Observable.FromEventPattern<EventHandler, EventArgs>(
                        handler => timer.Tick += handler,
                        handler => timer.Tick -= handler);
                    timer.Start();
                    var mergedSource = source.SelectMany(xs => xs.Do(
                        _ => { },
                        () => visualizerControl.BeginInvoke((Action)SequenceCompleted)));
                    return mergedSource
                        .Timestamp(HighResolutionScheduler.Default)
                        .Buffer(() => timerTick)
                        .Do(ShowBuffer)
                        .Finally(timer.Stop);
                });
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
