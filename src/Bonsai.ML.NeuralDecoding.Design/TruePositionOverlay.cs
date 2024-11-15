using Bonsai;
using Bonsai.Design;
using Bonsai.Design.Visualizers;
using System;
using System.Collections.Generic;
using OxyPlot.Series;
using OxyPlot;

[assembly: TypeVisualizer(typeof(Bonsai.ML.NeuralDecoding.Design.TruePositionOverlay),
    Target = typeof(MashupSource<Bonsai.ML.NeuralDecoding.Design.PosteriorVisualizer, TimeSeriesVisualizer>))]

namespace Bonsai.ML.NeuralDecoding.Design
{
    /// <summary>
    /// Class that overlays the true 
    /// </summary>
    public class TruePositionOverlay : DialogTypeVisualizer
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
                Title = "True Position",
                Color = OxyColors.Goldenrod
            };

            plot.Model.Series.Add(lineSeries);

            plot.Model.Updated += (sender, e) =>
            {
                plot.Model.DefaultYAxis.Title = "Position";
            };
            
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            var position = (double)value;
            if (position == double.NaN)
            {
                return;
            }

            data.Add(position);

            var currentCount = visualizer.CurrentCount;
            while (data.Count > currentCount)
            {
                data.RemoveAt(0);
            }
            lineSeries.Points.Clear();
            
            var count = data.Count;
            for (int i = 0; i < count; i++)
            {
                lineSeries.Points.Add(new DataPoint(currentCount - count + i, data[i]));
            }
        }

        /// <inheritdoc/>
        public override void Unload()
        {       
        }
    }
}