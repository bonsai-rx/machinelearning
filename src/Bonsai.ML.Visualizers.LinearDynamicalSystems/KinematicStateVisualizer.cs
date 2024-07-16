using Bonsai.Design;
using Bonsai;
using Bonsai.ML.LinearDynamicalSystems.Kinematics;
using Bonsai.ML.Visualizers.LinearDynamicalSystems;
using System.Collections.Generic;
using System;
using System.Windows.Forms;
using System.Reactive.Linq;
using System.Linq;
using System.Reactive;

[assembly: TypeVisualizer(typeof(KinematicStateVisualizer), Target = typeof(KinematicState))]

namespace Bonsai.ML.Visualizers.LinearDynamicalSystems
{

    /// <summary>
    /// Provides a type visualizer to display the state components of a Kalman Filter Kinematics model.
    /// </summary>
    public class KinematicStateVisualizer : MashupVisualizer
    {
        private TableLayoutPanel container;
        private int updateFrequency = 20;
        private bool resetAxes = true;
        private int rowCount = 3;
        private int columnCount = 2;

        internal List<StateComponentVisualizer> ComponentVisualizers { get; private set; } = new();

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
                ColumnCount = columnCount,
                RowCount = rowCount,
                Dock = DockStyle.Fill
            };

            for (int i = 0; i < container.RowCount; i++)
            {
                container.RowStyles.Add(new RowStyle(SizeType.Percent, 100f / rowCount));
            }

            for (int i = 0; i < container.ColumnCount; i++)
            {
                container.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f / columnCount));
            }

            for (int i = 0 ; i < rowCount; i++)
            {
                for (int j = 0; j < columnCount; j++)
                {
                    var StateComponentVisualizer = new StateComponentVisualizer() {
                        Label = Labels[i * columnCount + j]
                    };
                    StateComponentVisualizer.Load(provider);
                    container.Controls.Add(StateComponentVisualizer.Plot, j, i);
                    ComponentVisualizers.Add(StateComponentVisualizer);
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

            ComponentVisualizers[0].ShowDataBuffer(positionX, resetAxes);
            ComponentVisualizers[1].ShowDataBuffer(positionY, resetAxes);
            ComponentVisualizers[2].ShowDataBuffer(velocityX, resetAxes);
            ComponentVisualizers[3].ShowDataBuffer(velocityY, resetAxes);
            ComponentVisualizers[4].ShowDataBuffer(accelerationX, resetAxes);
            ComponentVisualizers[5].ShowDataBuffer(accelerationY, resetAxes);
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
            foreach (var componentVisualizer in ComponentVisualizers) 
            {
                componentVisualizer.Unload();
            }
            if (ComponentVisualizers.Count > 0) 
            {
                ComponentVisualizers.Clear();
                ComponentVisualizers = new();
            }
            if (!container.IsDisposed) container.Dispose();
        }
    }
}
