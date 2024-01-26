using System;
using System.Windows.Forms;
using OxyPlot;
using OxyPlot.Series;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers;
using Bonsai.ML.LinearDynamicalSystems;
using System.Drawing;
using OxyPlot.WindowsForms;
using OxyPlot.Axes;

[assembly: TypeVisualizer(typeof(EstimateWithUncertaintyVisualizerOxyPlot), Target = typeof(EstimateWithUncertainty))]

namespace Bonsai.ML.Visualizers
{
    // [TypeVisualizer(visualizer: typeof(EstimateWithUncertaintyVisualizerOxyPlot), TargetTypeName = "Bonsai.ML.LinearDynamicalSystems.EstimateWithUncertainty")]
    public class EstimateWithUncertaintyVisualizerOxyPlot : DialogTypeVisualizer
    {

        public EstimateWithUncertaintyVisualizerOxyPlot ()
        {
            Capacity = 10;
            Size = new Size(320, 240);
        }

        private EstimateTypes estimateType = EstimateTypes.X;

        public EstimateTypes EstimateType 
        {
            get => estimateType;
            set => estimateType = value;
        }

        public Size Size { get; set; }

        public int Capacity { get; set; }

        public double Min { get; set; }

        public double Max { get; set; }

        public bool AutoScale { get; set; }

        DateTime? _startTime;

        PlotView View;
        PlotModel Model;
        LineSeries Estimate;
        AreaSeries Uncertainty;
        ComboBox EstimateProperty;
        Label estimateLabel;

        public override void Load(IServiceProvider provider)
        {
            View = new PlotView
            {
                Size = Size,
                Dock = DockStyle.Fill,
            };

            Model = new PlotModel();

            Estimate = new LineSeries
            {
                Title = "Estimate",
                Color = OxyColors.Blue
            };

            Uncertainty = new AreaSeries
            {
                Title = "Uncertainty",
                Color = OxyColors.LightBlue,
                Fill = OxyColor.FromArgb(100, 173, 216, 230) // Light Blue with some transparency
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

            Model.Series.Add(Uncertainty);
            Model.Series.Add(Estimate);

            // Model.Axes[0].Minimum = 0;
            // Model.Axes[0].Maximum = 10;

            View.Model = Model;

            estimateLabel = new Label();
            estimateLabel.Text = "Estimate:";
            estimateLabel.AutoSize = true;
            estimateLabel.Location = new Point(-140, 10);
            estimateLabel.Anchor = AnchorStyles.Top | AnchorStyles.Right;

            EstimateProperty = new ComboBox();
            EstimateProperty.Location = new Point(-5, 5);
            EstimateProperty.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            EstimateProperty.DataSource = Enum.GetValues(typeof(EstimateTypes));
            EstimateProperty.SelectedIndexChanged += EstimateTypesChanged;

            // graph.GraphPane.YAxis.Scale.MaxAuto = true;
            // graph.GraphPane.YAxis.Scale.MinAuto = true;

            // graph.GraphPane.XAxis.Scale.Min = -1;
            // graph.GraphPane.XAxis.Scale.Max = 1;

            // graph.GraphPane.YAxis.Scale.Min = -1;
            // graph.GraphPane.YAxis.Scale.Max = 1;

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            if (visualizerService != null)
            {
                visualizerService.AddControl(EstimateProperty);
                visualizerService.AddControl(estimateLabel);
                visualizerService.AddControl(View);

                EstimateProperty.BringToFront();
                estimateLabel.BringToFront();
            }
        }

        public override void Show(object value)
        {
            EstimateWithUncertainty estimateWithUncertainty = (EstimateWithUncertainty)value;

            if (!_startTime.HasValue)
            {
                // Initialize startTime when the first data point arrives
                _startTime = DateTime.Now;
            }

            double estimate;
            double uncertainty;

            if (estimateType == EstimateTypes.X)
            {
                estimate = estimateWithUncertainty.X_state;
                uncertainty = estimateWithUncertainty.X_uncertainty;
            }
            else
            {
                estimate = estimateWithUncertainty.Y_state;
                uncertainty = estimateWithUncertainty.Y_uncertainty;
            }

            var time = (DateTime.Now - _startTime.Value).TotalSeconds;
            // var time = DateTimeAxis.ToDouble(dt);

            // Console.WriteLine($"dt: {dt}");
            // Console.WriteLine($"time: {time}");

            Estimate.Points.Add(new DataPoint(time, estimate));
            Uncertainty.Points.Add(new DataPoint(time, estimate + uncertainty));
            Uncertainty.Points2.Add(new DataPoint(time, estimate - uncertainty));

            var max_time = Math.Ceiling(time);
            var min_time = max_time - Capacity;

            // Console.WriteLine($"max_time: {max_time}");
            // Console.WriteLine($"min_time: {min_time}");

            if (min_time > 0)
            {
                // Model.Axes[0].Minimum = Estimate.Points[Estimate.Points.Count - Capacity].X;
                // Model.Axes[0].Maximum = (Math.Ceiling(time * 100) / 100) + 0.01;
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

        public enum EstimateTypes
        {
            X = 0,
            Y = 1
        }

        private void EstimateTypesChanged(object sender, EventArgs e)
        {
            string selectedItem = EstimateProperty.SelectedItem.ToString();
            EstimateType = (EstimateTypes)Enum.Parse(typeof(EstimateTypes), selectedItem);
            _startTime = null;
            Estimate.Points.Clear();
            Uncertainty.Points.Clear();
            Uncertainty.Points2.Clear();
            Model.Axes[0].Minimum = 0;
            Model.Axes[0].Maximum = Capacity;
            Model.InvalidatePlot(true);
        }
    }
}