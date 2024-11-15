using Bonsai;
using Bonsai.Design;
using System;
using System.Collections.Generic;
using OxyPlot.Series;
using OxyPlot;
using Bonsai.ML.Design;
using System.Windows.Forms;
using System.Reactive.Linq;
using System.Linq;

[assembly: TypeVisualizer(typeof(Bonsai.ML.NeuralDecoding.Design.PosteriorVisualizer),
    Target = typeof(Bonsai.ML.NeuralDecoding.Posterior))]

namespace Bonsai.ML.NeuralDecoding.Design
{
    /// <summary>
    /// Provides a mashup visualizer to display the posterior of the neural decoder.
    /// </summary>    
    public class PosteriorVisualizer : MashupVisualizer
    {
        private UnidimensionalArrayTimeSeriesVisualizer visualizer;
        private LineSeries lineSeries;
        private List<double> argMaxVals = new();
        private double[] valueRange = null;

        /// <summary>
        /// Gets the underlying heatmap plot.
        /// </summary>
        public HeatMapSeriesOxyPlotBase Plot => visualizer.Plot;
        
        /// <summary>
        /// Gets the capacity of the visualizer.
        /// </summary>
        public int Capacity => visualizer.Capacity;

        /// <summary>
        /// Gets the current count of data points.
        /// </summary>
        public int CurrentCount => visualizer.CurrentCount;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            visualizer = new UnidimensionalArrayTimeSeriesVisualizer();
            visualizer.Load(provider);

            lineSeries = new LineSeries()
            {
                Color = OxyColors.SkyBlue
            };
            visualizer.Plot.Model.Series.Add(lineSeries);

            base.Load(provider);
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            Posterior posterior = (Posterior)value;
            if (posterior == null)
            {
                return;
            }

            if (valueRange == null)
            {
                valueRange = posterior.ValueRange;
            }

            var data = posterior.Data;
            var argMax = posterior.ArgMax;
            var positionRange = posterior.PositionRange;
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
            base.Unload();
        }

        /// <inheritdoc/>
        public override IObservable<object> Visualize(IObservable<IObservable<object>> source, IServiceProvider provider)
        {
            if (provider.GetService(typeof(IDialogTypeVisualizerService)) is not Control visualizerControl)
            {
                return source;
            }

            var mergedSource = source.SelectMany(xs => xs
                .ObserveOn(visualizerControl)
                .Do(value => Show(value)));

            var mashupSourceStreams = Observable.Merge(
                MashupSources.Select(mashupSource =>
                    mashupSource.Visualizer.Visualize(mashupSource.Source.Output, provider)));

            return Observable.Merge(mergedSource, mashupSourceStreams);
        }
    }
}
