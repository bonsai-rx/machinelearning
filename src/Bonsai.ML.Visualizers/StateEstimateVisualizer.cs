using Bonsai;
using System;
using Bonsai.Design;
using Bonsai.Design.Visualizers;
using System.Windows.Forms;
using Bonsai.ML.Visualizers;
using Bonsai.ML.LinearDynamicalSystems;

[assembly: TypeVisualizer(typeof(KinematicsVisualizer), Target = typeof(Kinematics))]

namespace Bonsai.ML.Visualizers
{
    // [TypeVisualizer(typeof(KinematicsVisualizer), Target = typeof(StateEstimate))]
    public class KinematicsVisualizer : DialogTypeVisualizer
    {
        GraphControl graph = null;

        public override void Load(IServiceProvider provider)
        {
            graph = new GraphControl();
            graph.Dock = DockStyle.Fill;

            graph.GraphPane.YAxis.Scale.MaxAuto = false;
            graph.GraphPane.YAxis.Scale.MinAuto = false;

            graph.GraphPane.XAxis.Scale.Min = -1;
            graph.GraphPane.XAxis.Scale.Max = 1;

            graph.GraphPane.YAxis.Scale.Min = -1;
            graph.GraphPane.YAxis.Scale.Max = 1;

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
            if (visualizerService != null)
            {
                visualizerService.AddControl(graph);
            }
        }

        public override void Show(object value)
        {
            if (!graph.IsDisposed)
            {
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
    }
}
