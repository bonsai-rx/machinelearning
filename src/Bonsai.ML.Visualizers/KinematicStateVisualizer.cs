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

[assembly: TypeVisualizer(typeof(KinematicStateVisualizer), Target = typeof(KinematicState))]

namespace Bonsai.ML.Visualizers
{

    /// <summary>
    /// Provides a type visualizer to display the state components of a Kalman Filter Kinematics model.
    /// </summary>
    public class KinematicStateVisualizer : MashupVisualizer
    {

        private int selectedStateIndex = 0;
        private int selectedKinematicIndex = 0;
        private DateTime? _startTime;
        private TimeSpan updateFrequency = TimeSpan.FromSeconds(1/30);
        private LineSeries lineSeries;
        private AreaSeries areaSeries;

        /// <summary>
        /// The selected state component property of the Kinematic State object being visualized.
        /// </summary>
        [XmlIgnore()]
        public PropertyInfo stateComponentProperty { get; private set; }

        /// <summary>
        /// The selected kinematic component property of the Kinematic State object being visualized.
        /// </summary>
        [XmlIgnore()]
        public PropertyInfo kinematicComponentProperty { get; private set; }

        /// <summary>
        /// The underlying plot used for visualization.
        /// </summary>
        // [XmlIgnore()]
        internal TimeSeriesOxyPlotBase Plot { get; private set; }

        /// <summary>
        /// The selected index of the state component to be visualized
        /// </summary>
        public int SelectedStateIndex { get => selectedStateIndex; set => selectedStateIndex = value; }

        /// <summary>
        /// The selected index of the kinematic component to be visualized
        /// </summary>
        public int SelectedKinematicIndex { get => selectedKinematicIndex; set => selectedKinematicIndex = value; }

        /// <summary>
        /// Size of the window when loaded
        /// </summary>
        public Size Size { get; set; } = new Size(320, 240);

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
                Dock = DockStyle.Fill,
                StartTime = DateTime.Now,
                Capacity = Capacity,
                Size = Size,
                BufferData = BufferData
            };

            lineSeries = Plot.AddNewLineSeries("Mean");
            areaSeries = Plot.AddNewAreaSeries("Variance");

            Plot.ResetLineSeries(lineSeries);
            Plot.ResetAreaSeries(areaSeries);
            Plot.ResetAxes();

            List<string> stateComponents = KinematicsHelper.GetStateComponents();
            List<string> kinematicComponents = KinematicsHelper.GetKinematicComponents();

            Plot.AddComboBoxWithLabel("State component:", stateComponents, selectedStateIndex, StateComponentChanged);
            Plot.AddComboBoxWithLabel("Kinematic component:", kinematicComponents, selectedKinematicIndex, KinematicComponentChanged);

            stateComponentProperty = typeof(KinematicComponent).GetProperty(stateComponents[selectedStateIndex]);
            kinematicComponentProperty = typeof(KinematicState).GetProperty(kinematicComponents[selectedKinematicIndex]);

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            if (visualizerService != null)
            {
                visualizerService.AddControl(Plot);
            }
            _startTime = null;

            base.Load(provider);
        }


        /// <inheritdoc/>
        public override void Show(object value)
        {
            var time = DateTime.Now;
            if (!_startTime.HasValue)
            {
                _startTime = time;
                Plot.StartTime = _startTime.Value;
                Plot.ResetAxes();
            }

            KinematicState kinematicState = (KinematicState)value;
            KinematicComponent kinematicComponent = (KinematicComponent)kinematicComponentProperty.GetValue(kinematicState);
            StateComponent stateComponent = (StateComponent)stateComponentProperty.GetValue(kinematicComponent);

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

            if (MashupSources.Count == 0) Plot.SetAxes(minTime: time.AddSeconds(-Capacity), maxTime: time);

        }

        /// <inheritdoc/>
        public override IObservable<object> Visualize(IObservable<IObservable<object>> source, IServiceProvider provider)
        {
            var mashupSourceStreams = MashupSources.Select(mashupSource =>
                mashupSource.Visualizer.Visualize(mashupSource.Source.Output, provider)
                    .Do(value => mashupSource.Visualizer.Show(value)));

            var mergedMashupSources = Observable.Merge(mashupSourceStreams);

            var processedSource = base.Visualize(source, provider);

            return Observable.Merge(mergedMashupSources, processedSource)
                .Sample(updateFrequency)
                .Do(_ => Plot.UpdatePlot());
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

        private void StateComponentChanged(object sender, EventArgs e)
        {
            ToolStripComboBox comboBox = sender as ToolStripComboBox;
            selectedStateIndex = comboBox.SelectedIndex;
            var selectedName = comboBox.SelectedItem.ToString();
            stateComponentProperty = typeof(KinematicComponent).GetProperty(selectedName);
            _startTime = null;

            Plot.ResetLineSeries(lineSeries);
            Plot.ResetAreaSeries(areaSeries);
            Plot.ResetAxes();
        }

        private void KinematicComponentChanged(object sender, EventArgs e)
        {
            ToolStripComboBox comboBox = sender as ToolStripComboBox;
            selectedKinematicIndex = comboBox.SelectedIndex;
            var selectedName = comboBox.SelectedItem.ToString();
            kinematicComponentProperty = typeof(KinematicState).GetProperty(selectedName);
            _startTime = null;

            Plot.ResetLineSeries(lineSeries);
            Plot.ResetAreaSeries(areaSeries);
            Plot.ResetAxes();
        }
    }
}
