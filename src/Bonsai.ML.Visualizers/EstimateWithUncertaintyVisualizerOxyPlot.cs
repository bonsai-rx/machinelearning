using System;
using System.Windows.Forms;
using OxyPlot;
using OxyPlot.Series;
using Bonsai;
using Bonsai.Design;
using Bonsai.Design.Visualizers;
using Bonsai.ML.Visualizers;
using Bonsai.ML.LinearDynamicalSystems;
using System.Drawing;
using System.Linq;
using System.Collections.Generic;
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
        }

        private EstimateTypes estimateType = EstimateTypes.X;

        public EstimateTypes EstimateType 
        {
            get => estimateType;
            set => estimateType = value;
        }

        public int Capacity { get; set; }

        public double Min { get; set; }

        public double Max { get; set; }

        public bool AutoScale { get; set; }

        DateTime? _startTime;

        PlotView View;
        PlotModel Model;
        LineSeries Estimate;
        AreaSeries Uncertainty;

        public override void Load(IServiceProvider provider)
        {
            View = new PlotView
            {
                Size = new Size(320, 240),
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
                Minimum = _startTime.HasValue & Convert.ToDouble(_startTime.Value) > Capacity ? Convert.ToDouble(_startTime.Value) - Capacity : 0,
                Maximum = _startTime.HasValue & Convert.ToDouble(_startTime.Value) > Capacity ? Convert.ToDouble(_startTime.Value) : Capacity
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

            // graph.GraphPane.YAxis.Scale.MaxAuto = true;
            // graph.GraphPane.YAxis.Scale.MinAuto = true;

            // graph.GraphPane.XAxis.Scale.Min = -1;
            // graph.GraphPane.XAxis.Scale.Max = 1;

            // graph.GraphPane.YAxis.Scale.Min = -1;
            // graph.GraphPane.YAxis.Scale.Max = 1;

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            if (visualizerService != null)
            {
                visualizerService.AddControl(View);
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
            Console.WriteLine($"time: {time}");

            Estimate.Points.Add(new DataPoint(time, estimate));
            Uncertainty.Points.Add(new DataPoint(time, estimate + uncertainty));
            Uncertainty.Points2.Add(new DataPoint(time, estimate - uncertainty));

            var max_time = Math.Ceiling(time);
            var min_time = max_time - Capacity;

            Console.WriteLine($"max_time: {max_time}");
            Console.WriteLine($"min_time: {min_time}");

            if (Estimate.Points.Count > Capacity)
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
    }
}