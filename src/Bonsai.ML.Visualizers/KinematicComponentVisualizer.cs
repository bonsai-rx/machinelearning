using System;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Reflection;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers;
using Bonsai.ML.LinearDynamicalSystems;
using Bonsai.ML.LinearDynamicalSystems.Kinematics;
using System.Drawing;
using System.Reactive;
using OxyPlot.Series;

[assembly: TypeVisualizer(typeof(KinematicComponentVisualizer), Target = typeof(KinematicComponent))]

namespace Bonsai.ML.Visualizers
{
    /// <summary>
    /// Provides a type visualizer to display the state components of a Kalman Filter kinematics model.
    /// </summary>
    public class KinematicComponentVisualizer : BufferedVisualizer
    {

        private PropertyInfo stateComponentProperty;

        private int selectedIndex = 0;

        private DateTime? _startTime;

        private TimeSeriesOxyPlotBase Plot;

        private LineSeries lineSeries;

        private AreaSeries areaSeries;

        /// <summary>
        /// The selected index of the state component to be visualized
        /// </summary>
        public int SelectedIndex { get => selectedIndex; set => selectedIndex = value; }

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
                Size = Size,
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

            Plot.AddComboBoxWithLabel("State component:", KinematicsHelper.GetStateComponents(), selectedIndex, ComponentChanged);

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
                // Plot.ResetSeries();

            }

            KinematicComponent kinematicComponent = (KinematicComponent)value;
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

        private void ComponentChanged(object sender, EventArgs e)
        {
            ToolStripComboBox comboBox = sender as ToolStripComboBox;
            selectedIndex = comboBox.SelectedIndex;
            var selectedName = comboBox.SelectedItem.ToString();
            stateComponentProperty = typeof(KinematicComponent).GetProperty(selectedName);
            _startTime = null;

            Plot.ResetAxes();
        }
    }
}
