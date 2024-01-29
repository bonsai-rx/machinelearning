using System;
using System.Windows.Forms;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reflection;
using OxyPlot;
using OxyPlot.Series;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers;
using Bonsai.ML.LinearDynamicalSystems;
using System.Drawing;
using OxyPlot.WindowsForms;
using OxyPlot.Axes;

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
        private string stateComponentName;

        private int stateComponentSelectedIndex = 0;
        public int StateComponentSelectedIndex { get => stateComponentSelectedIndex; set => stateComponentSelectedIndex = value; }

        public Size Size { get; set; }

        public int Capacity { get; set; }

        DateTime? _startTime;

        PlotView View;
        PlotModel Model;
        LineSeries Mean;
        AreaSeries Variance;
        ComboBox StateComponentComboBox;
        Label StateComponentLabel;

        public override void Load(IServiceProvider provider)
        {
            View = new PlotView
            {
                Size = Size,
                Dock = DockStyle.Fill,
            };

            Model = new PlotModel();

            Mean = new LineSeries
            {
                Title = "Mean",
                Color = OxyColors.Blue
            };

            Variance = new AreaSeries
            {
                Title = "Variance",
                Color = OxyColors.LightBlue,
                Fill = OxyColor.FromArgb(100, 173, 216, 230)
            };

            Model.Axes.Add(new LinearAxis {
                Position = AxisPosition.Bottom,
                Title = "Time",
                StringFormat = "0",
                MajorGridlineStyle = LineStyle.Solid,
                MinorGridlineStyle = LineStyle.Dot,
                MinimumMajorStep = 1,
                MinimumMinorStep = 0.5,
                Minimum = _startTime.HasValue ? Convert.ToDouble(_startTime.Value) - Capacity : 0,
                Maximum = _startTime.HasValue ? Convert.ToDouble(_startTime.Value) : Capacity
            });

            Model.Axes.Add(new LinearAxis {
                Position = AxisPosition.Left,
                Title = "Estimate"
            });

            Model.Series.Add(Mean);
            Model.Series.Add(Variance);

            View.Model = Model;

            StateComponentLabel = new Label
            {
                Text = "State component:",
                AutoSize = true,
                Location = new Point(-140, 10),
                Anchor = AnchorStyles.Top | AnchorStyles.Right
            };

            StateComponentComboBox = new ComboBox
            {
                Location = new Point(-5, 5),
                Anchor = AnchorStyles.Top | AnchorStyles.Right,
                DataSource = GetStateComponents()
            };

            StateComponentComboBox.SelectedIndexChanged += InitializeSelection;

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            if (visualizerService != null)
            {
                visualizerService.AddControl(StateComponentComboBox);
                visualizerService.AddControl(StateComponentLabel);
                visualizerService.AddControl(View);

                StateComponentComboBox.BringToFront();
                StateComponentLabel.BringToFront();
            }
        }

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

            Mean.Points.Add(new DataPoint(time, mean));
            Variance.Points.Add(new DataPoint(time, mean + variance));
            Variance.Points2.Add(new DataPoint(time, mean - variance));

            var max_time = Math.Ceiling(time);
            var min_time = max_time - Capacity;

            if (min_time > 0)
            {
                Model.Axes[0].Minimum = min_time;
                Model.Axes[0].Maximum = max_time;
            }

            Model.InvalidatePlot(true);
        }

        public override void Unload()
        {
            _startTime = null;
            if (!View.IsDisposed)
            {
                View.Dispose();
            }
        }

        private void InitializeSelection(object sender, EventArgs e)
        {
            StateComponentComboBox.SelectedIndex = stateComponentSelectedIndex;
            StateComponentComboBox.SelectedIndexChanged -= InitializeSelection;
            StateComponentComboBox.SelectedIndexChanged += ComponentChanged;
        }

        private void ComponentChanged(object sender, EventArgs e)
        {
            var selectedName = StateComponentComboBox.SelectedItem.ToString();
            if (selectedName != stateComponentName)
            {
                stateComponentSelectedIndex = StateComponentComboBox.SelectedIndex;
                stateComponentName = selectedName;
                stateComponentProperty = typeof(KinematicComponent).GetProperty(stateComponentName);
                _startTime = null;
                Mean.Points.Clear();
                Variance.Points.Clear();
                Variance.Points2.Clear();
                Model.Axes[0].Minimum = 0;
                Model.Axes[0].Maximum = Capacity;
                Model.InvalidatePlot(true);
            }
        }
    }
}