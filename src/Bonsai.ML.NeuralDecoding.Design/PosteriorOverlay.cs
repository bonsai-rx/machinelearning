using Bonsai;
using Bonsai.Design;
using Bonsai.Design.Visualizers;
using System;
using System.Collections.Generic;
using OxyPlot.Series;
using OxyPlot;
using Bonsai.ML.Design;

[assembly: TypeVisualizer(typeof(Bonsai.ML.NeuralDecoding.Design.PosteriorOverlay),
    Target = typeof(MashupSource<Bonsai.ML.NeuralDecoding.Design.PosteriorVisualizer, TimeSeriesVisualizer>))]

namespace Bonsai.ML.NeuralDecoding.Design
{
    /// <summary>
    /// 
    /// </summary>
    public class PosteriorOverlay : DialogTypeVisualizer
    {
        private PosteriorVisualizer visualizer;
        private LineSeries lineSeries;
        private List<double> data = new();

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            var service = provider.GetService(typeof(MashupVisualizer));
            visualizer = (PosteriorVisualizer)service;
            var plot = visualizer.Plot;

            lineSeries = new LineSeries()
            {
                Color = OxyColors.Goldenrod
            };
            plot.Model.Series.Add(lineSeries);
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            var position = (double)value;
            var capacity = visualizer.Capacity;
            while (data.Count >= capacity)
            {
                data.RemoveAt(0);
            }
            data.Add(position);
            lineSeries.Points.Clear();
            var count = data.Count;
            for (int i = 0; i < count; i++)
            {
                lineSeries.Points.Add(new DataPoint(i, data[i]));
            }
        }

        /// <inheritdoc/>
        public override void Unload()
        {       
        }
    }
}