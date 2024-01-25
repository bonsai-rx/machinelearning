using System;
using System.Windows.Forms;
using ZedGraph;
using Bonsai;
using Bonsai.Design;
using Bonsai.Design.Visualizers;
using Bonsai.ML.Visualizers;
using Bonsai.ML.LinearDynamicalSystems;
using System.Drawing;
using System.Linq;
using System.Collections.Generic;

[assembly: TypeVisualizer(typeof(EstimateWithUncertaintyVisualizer), Target = typeof(EstimateWithUncertainty))]

namespace Bonsai.ML.Visualizers
{
    // [TypeVisualizer(visualizer: typeof(EstimateWithUncertaintyVisualizer), TargetTypeName = "Bonsai.ML.LinearDynamicalSystems.EstimateWithUncertainty")]
    public class EstimateWithUncertaintyVisualizer : DialogTypeVisualizer
    {

        private EstimateTypes estimateType = EstimateTypes.X;
        public EstimateTypes EstimateType 
        {
            get => estimateType;
            set => estimateType = value;
        }
        GraphControl graph;
        PointPairList pointPairs = new PointPairList();
        List<PointD> stdPoints = new List<PointD>();

        public override void Load(IServiceProvider provider)
        {
            graph = new GraphControl();
            graph.Dock = DockStyle.Fill;

            graph.GraphPane.YAxis.Scale.MaxAuto = true;
            graph.GraphPane.YAxis.Scale.MinAuto = true;

            // graph.GraphPane.XAxis.Scale.Min = -1;
            // graph.GraphPane.XAxis.Scale.Max = 1;

            // graph.GraphPane.YAxis.Scale.Min = -1;
            // graph.GraphPane.YAxis.Scale.Max = 1;

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            if (visualizerService != null)
            {
                visualizerService.AddControl(graph);
            }
        }

        public override void Show(object value)
        {
            EstimateWithUncertainty estimateWithUncertainty = (EstimateWithUncertainty)value;

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

            XDate time = DateTime.Now;

            pointPairs.Add(time, estimate);
            stdPoints.Add(new PointD(time, estimate + uncertainty));
            stdPoints.Add(new PointD(time, estimate - uncertainty));

            graph.GraphPane.AddCurve(
                "", 
                pointPairs,
                Color.SkyBlue
            );

            var color = Color.FromArgb(100, Color.IndianRed);

            // var poly = new PolyObj
            // {
            //     Points = stdPoints.ToArray(),
            //     Border = new Border(Color.Transparent, 0),
            //     Fill = new Fill(color)
            // };

            var poly = new PolyObj
            {
                Points = stdPoints.ToArray(),
                Border = new Border(Color.Transparent, 0)
            };

            graph.GraphPane.GraphObjList.Add(poly);

            // LineItem curve = new LineItem("", X, Ys[i], Color.Red, SymbolType.None);
            // curve.Line.IsAntiAlias = true;
            // curve.Line.IsOptimizedDraw = true;
            if (!graph.IsDisposed)
            {
                graph.AxisChange();
                graph.Invalidate();
            }
        }

        public override void Unload()
        {
            if (!graph.IsDisposed)
            {
                graph.Dispose();
            }
        }

        public enum EstimateTypes
        {
            X = 0,
            Y = 1
        }
    }
}