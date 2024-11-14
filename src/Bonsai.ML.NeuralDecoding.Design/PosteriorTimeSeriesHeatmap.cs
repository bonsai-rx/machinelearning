using Bonsai;
using Bonsai.Design;
using System;
using System.Collections.Generic;
using OxyPlot.Series;
using OxyPlot;
using Bonsai.ML.Design;

[assembly: TypeVisualizer(typeof(Bonsai.ML.NeuralDecoding.Design.PosteriorTimeSeriesHeatmap),
    Target = typeof(Bonsai.ML.NeuralDecoding.Posterior))]

namespace Bonsai.ML.NeuralDecoding.Design
{
    public class PosteriorTimeSeriesHeatmap : DialogTypeVisualizer
    {
        private UnidimensionalArrayTimeSeriesVisualizer visualizer;
        private LineSeries lineSeries;
        private List<double> argMaxVals = new();

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            visualizer = new UnidimensionalArrayTimeSeriesVisualizer();
            visualizer.Load(provider);

            lineSeries = new LineSeries();
            visualizer.Plot.Model.Series.Add(lineSeries);
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            Posterior posterior = (Posterior)value;
            var data = posterior.Data;
            var argMax = posterior.ArgMax;
            var capacity = visualizer.Capacity;
            while (argMaxVals.Count >= capacity)
            {
                argMaxVals.RemoveAt(0);
            }
            argMaxVals.Add(argMax);
            lineSeries.Points.Clear();
            var count = argMaxVals.Count;
            for (int i = 0; i < count; i++)
            {
                lineSeries.Points.Add(new DataPoint(i, argMaxVals[i]));
            }
            visualizer.Show(data);
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            visualizer.Unload();
        }
    }
}
