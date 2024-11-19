using Bonsai;
using Bonsai.Design;
using Bonsai.Design.Visualizers;
using System;
using System.Collections.Generic;
using OxyPlot.Series;
using OxyPlot;
using Bonsai.ML.Design;

[assembly: TypeVisualizer(typeof(Bonsai.ML.NeuralDecoder.Design.TruePositionOverlay),
    Target = typeof(MashupSource<Bonsai.ML.NeuralDecoder.Design.PosteriorVisualizer, TimeSeriesVisualizer>))]

namespace Bonsai.ML.NeuralDecoder.Design
{
    /// <summary>
    /// Class that overlays the true 
    /// </summary>
    public class TruePositionOverlay : DialogTypeVisualizer
    {
        private PosteriorVisualizer visualizer;
        private LineSeries lineSeries;
        private List<double> data = new();
        private string defaultYAxisTitle;
        private HeatMapSeriesOxyPlotBase plot;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            var service = provider.GetService(typeof(MashupVisualizer));
            visualizer = (PosteriorVisualizer)service;
            plot = visualizer.Plot;

            lineSeries = new LineSeries()
            {
                Title = "True Position",
                Color = OxyColors.Goldenrod
            };

            plot.Model.Series.Add(lineSeries);

            plot.Model.Updated += (sender, e) =>
            {
                defaultYAxisTitle = plot.Model.DefaultYAxis.Title;
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
            var valueRange = visualizer.ValueRange;
            var valueCenters = visualizer.ValueCenters;

            while (data.Count > currentCount)
            {
                data.RemoveAt(0);
            }
            lineSeries.Points.Clear();
            
            var count = data.Count;
            for (int i = 0; i < count; i++)
            {
                var closestIndex = Array.BinarySearch(valueRange, data[i]);
                if (closestIndex < 0)
                {
                    closestIndex = ~closestIndex;
                }
                lineSeries.Points.Add(new DataPoint(currentCount - count + i, valueCenters[closestIndex]));
            }
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            plot.Model.DefaultYAxis.Title = defaultYAxisTitle;
        }
    }
}