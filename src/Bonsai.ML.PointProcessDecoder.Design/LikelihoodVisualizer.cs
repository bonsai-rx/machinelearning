using System;
using System.Reactive.Linq;
using System.Reactive;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Linq;

using Bonsai;
using Bonsai.Expressions;
using Bonsai.Design;
using Bonsai.ML.Design;

using OxyPlot.Series;

using static TorchSharp.torch;

using PointProcessDecoder.Core;
using TorchSharp;
using Bonsai.Dag;
using System.Linq.Expressions;

[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.LikelihoodVisualizer), 
    Target = typeof(Bonsai.ML.PointProcessDecoder.Decode))]

namespace Bonsai.ML.PointProcessDecoder.Design
{
    /// <summary>
    /// Visualizer for the likelihood of a point process model.
    /// </summary>
    public class LikelihoodVisualizer : MashupVisualizer, IDecoderVisualizer
    {
        private MultidimensionalArrayVisualizer _visualizer;

        /// <summary>
        /// Gets the underlying heatmap plot.
        /// </summary>
        public HeatMapSeriesOxyPlotBase Plot => _visualizer.Plot;

        private int _capacity = 100;
        /// <summary>
        /// Gets or sets the integer value that determines how many data points should be shown along the x axis if the posterior is a 1D tensor.
        /// </summary>
        public int Capacity 
        { 
            get => _capacity;
            set 
            {
                _capacity = value;
            } 
        }

        /// <summary>
        /// Gets or sets the minimum value of the likelihood.
        /// </summary>
        public double? ValueMin { get; set; } = null;

        /// <summary>
        /// Gets or sets the maximum value of the likelihood.
        /// </summary>
        public double? ValueMax { get; set; } = null;

        private double[,] _data = null;
        private long _stateSpaceWidth;
        private long _stateSpaceHeight;
        private double[] _stateSpaceMin = null;
        private double[] _stateSpaceMax = null;
        private string _modelName;
        private ILikelihood _likelihood;
        private Tensor[] _intensities;
        private IObservable<IObservable<object>> _inputSource;
        
        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            Node<ExpressionBuilder, ExpressionBuilderArgument> visualizerNode = null;
            var expressionBuilderGraph = (ExpressionBuilderGraph)provider.GetService(typeof(ExpressionBuilderGraph));
            var typeVisualizerContext = (ITypeVisualizerContext)provider.GetService(typeof(ITypeVisualizerContext));
            if (expressionBuilderGraph != null && typeVisualizerContext != null)
            {
                visualizerNode = (from node in expressionBuilderGraph
                    where node.Value == typeVisualizerContext.Source
                    select node).FirstOrDefault();
            }

            if (visualizerNode == null)
            {
                throw new InvalidOperationException("The visualizer node is invalid.");
            }

            var inspector = (InspectBuilder)expressionBuilderGraph
                .Predecessors(visualizerNode)
                .First(p => !p.Value.IsBuildDependency())
                .Value;

            _inputSource = inspector.Output;

            if (ExpressionBuilder.GetWorkflowElement(visualizerNode.Value) is not Decode decodeNode)
            {
                throw new InvalidOperationException("The decode node is invalid.");
            }

            _modelName = decodeNode.Name;

            _visualizer = new MultidimensionalArrayVisualizer()
            {
                PaletteSelectedIndex = 1,
                RenderMethodSelectedIndex = 0
            };
            
            _visualizer.Load(provider);

            var capacityLabel = new ToolStripLabel
            {
                Text = "Capacity: ",
                AutoSize = true
            };

            var capacityValue = new ToolStripTextBox
            {
                Text = Capacity.ToString(),
                AutoSize = true
            };

            capacityValue.TextChanged += (sender, e) => 
            {
                if (int.TryParse(capacityValue.Text, out int capacity))
                {
                    Capacity = capacity;
                }
            };

            _visualizer.Plot.VisualizerPropertiesDropDown.DropDownItems.AddRange([
                capacityLabel,
                capacityValue
            ]);

            base.Load(provider);
        }

        private bool UpdateModel()
        {
            PointProcessModel model;

            try
            {
                model = PointProcessModelManager.GetModel(_modelName);
            } catch {
                return false;
            }

            if (model == null)
            {
                return false;
            }

            _stateSpaceWidth = model.StateSpace.Shape[0];
            _stateSpaceHeight = model.StateSpace.Shape[1];

            _stateSpaceMin ??= [.. model.StateSpace.Points
                .min(dim: 0)
                .values
                .to_type(ScalarType.Float64)
                .data<double>()
            ];

            _stateSpaceMax ??= [.. model.StateSpace.Points
                .max(dim: 0)
                .values
                .to_type(ScalarType.Float64)
                .data<double>()
            ];

            _likelihood = model.Likelihood;
            _intensities = model.Encoder.Intensities;

            return true;
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            Tensor inputs = (Tensor)value;
            Tensor likelihood = _likelihood.Likelihood(inputs, _intensities);

            if (likelihood.Dimensions == 2) {
                likelihood = likelihood
                    .mean([0]);
            }
            
            _data = (double[,])likelihood
                .to_type(ScalarType.Float64)
                .reshape([_stateSpaceWidth, _stateSpaceHeight])
                .data<double>()
                .ToNDArray();


            _visualizer.Plot.UpdateHeatMapSeries(
                _stateSpaceMin[0],
                _stateSpaceMax[0],
                _stateSpaceMin[1],
                _stateSpaceMax[1],
                _data
            );

            _visualizer.Plot.UpdatePlot();
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            _visualizer.Unload();
            base.Unload();
        }

        /// <inheritdoc/>
        public override IObservable<object> Visualize(IObservable<IObservable<object>> source, IServiceProvider provider)
        {
            if (provider.GetService(typeof(IDialogTypeVisualizerService)) is not Control visualizerControl)
            {
                return source;
            }

            var colorCycler = new OxyColorPresetCycle();

            var timer = Observable.Interval(
                TimeSpan.FromMilliseconds(100),
                HighResolutionScheduler.Default
            );

            var mergedSource = _inputSource.SelectMany(xs => 
                xs.Buffer(timer)
                    .Where(buffer => buffer.Count > 0)
                    .Sample(source.Merge())
                    .Do(buffer => {
                        if (!UpdateModel())
                        {
                            return;
                        }
                        ValueMin = _visualizer.Plot.ValueMin;
                        ValueMax = _visualizer.Plot.ValueMax;
                        Show(buffer.LastOrDefault());
                    }));

            var mashupSourceStreams = Observable.Merge(
                MashupSources.Select(mashupSource =>
                    mashupSource.Source.Output.SelectMany(xs => {
                        var color = colorCycler.Next();
                        var visualizer = mashupSource.Visualizer as Point2DOverlay;
                        visualizer.Color = color;
                        return xs.Buffer(timer)
                            .Where(buffer => buffer.Count > 0)
                            .Do(buffer => visualizer.Show(buffer.LastOrDefault()));
            })));

            return Observable.Merge(mergedSource, mashupSourceStreams);
        }
    }
}