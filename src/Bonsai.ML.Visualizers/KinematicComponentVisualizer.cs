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

[assembly: TypeVisualizer(typeof(KinematicComponentVisualizer), Target = typeof(KinematicComponent))]

namespace Bonsai.ML.Visualizers
{
    public class KinematicComponentVisualizer : DialogTypeVisualizer
    {

        public KinematicComponentVisualizer ()
        {
            Capacity = 10;
            Size = new Size(320, 240);
        }

        private PropertyInfo stateComponentProperty;

        private int selectedIndex = 0;

        /// <summary>
        /// The selected index of the state component to be visualized
        /// </summary>
        public int SelectedIndex { get => selectedIndex; set => selectedIndex = value; }

        /// <summary>
        /// Size of the window when loaded
        /// </summary>
        public Size Size { get; set; }

        /// <summary>
        /// Capacity or length of time shown along the x axis of the plot during automatic updating
        /// </summary>
        public int Capacity { get; set; }

        DateTime? _startTime;

        TimeSeriesOxyPlotBase Plot;

        public override void Load(IServiceProvider provider)
        {
            var stateComponents = GetStateComponents();
            stateComponentProperty = typeof(KinematicComponent).GetProperty(stateComponents[selectedIndex]);

            Plot = new TimeSeriesOxyPlotBase(
                _lineSeriesName: "Mean",
                _areaSeriesName: "Variance",
                _dataSource: stateComponents,
                _selectedIndex: selectedIndex
            )
            {
                Size = Size,
                StartTime = DateTime.Now,
                Capacity = Capacity,
                Dock = DockStyle.Fill
            };

            Plot.ResetSeries();

            Plot.ComboBoxValueChanged += ComponentChanged;

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            if (visualizerService != null)
            {
                visualizerService.AddControl(Plot);
            }
        }

        public override void Show(object value)
        {
            if (!_startTime.HasValue)
            {
                _startTime = DateTime.Now;
            }

            KinematicComponent kinematicComponent = (KinematicComponent)value;
            StateComponent stateComponent = (StateComponent)stateComponentProperty.GetValue(kinematicComponent);
            double mean = stateComponent.Mean;
            double variance = stateComponent.Variance;

            var time = (DateTime.Now - _startTime.Value).TotalSeconds;

            Plot.AddToLineSeries(
                time: time,
                mean: mean
            );

            Plot.AddToAreaSeries(
                time: time,
                mean: mean,
                variance: variance
            );

            var maxTime = Math.Ceiling(time);
            var minTime = maxTime - Capacity;

            if (minTime > 0)
            {
                Plot.SetAxes(minTime: minTime, maxTime: maxTime);
            }

            Plot.Update();
        }

        /// <summary>
        /// Gets the names of the state components defined in the kinematic component class
        /// </summary>
        private List<string> GetStateComponents()
        {
            List<string> stateComponents = new List<string>();

            foreach (PropertyInfo property in typeof(KinematicComponent).GetProperties())
            {
                if (property.PropertyType == typeof(StateComponent))
                {
                    stateComponents.Add(property.Name);
                }
            }

            return stateComponents;
        }

        public override void Unload()
        {
            _startTime = null;
            if (!Plot.IsDisposed)
            {
                Plot.Dispose();
            }
        }

        /// <summary>
        /// Callback function to update the visualizer when the selected component has changed
        /// </summary>
        private void ComponentChanged(object sender, EventArgs e)
        {
            var comboBox = Plot.ComboBox;
            if (comboBox.SelectedIndex != selectedIndex)
            {
                selectedIndex = comboBox.SelectedIndex;
                var selectedName = comboBox.SelectedItem.ToString();
                stateComponentProperty = typeof(KinematicComponent).GetProperty(selectedName);
                _startTime = null;

                Plot.ResetSeries();
                Plot.Update();
            }
        }
    }
}