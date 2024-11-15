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
        private double[] positionRange = null;

        /// <summary>
        /// Gets the underlying heatmap plot.
        /// </summary>
        public HeatMapSeriesOxyPlotBase Plot => visualizer.Plot;
        
        /// <summary>
        /// Gets the 
        /// </summary>
        public int Capacity => visualizer.Capacity;

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
        }

        /// <inheritdoc/>
        public override IObservable<object> Visualize(IObservable<IObservable<object>> source, IServiceProvider provider)
        {
            if (provider.GetService(typeof(IDialogTypeVisualizerService)) is not Control visualizerControl)
            {
                return source;
            }

            return Observable.Merge(source.SelectMany(xs => xs.Do(
                        value => Show(value),
                        () => visualizerControl.BeginInvoke(SequenceCompleted))), 
                        Observable.Merge(MashupSources.Select(
                            mashupSource => mashupSource.Visualizer.Visualize(mashupSource.Source.Output, provider)
                            .Do(value => mashupSource.Visualizer.Show(value)))));
        }
    }
}
