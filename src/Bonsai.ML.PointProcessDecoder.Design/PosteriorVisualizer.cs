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
using PointProcessDecoder.Core.Decoder;

[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.PosteriorVisualizer), 
    Target = typeof(Bonsai.ML.PointProcessDecoder.Decode))]
[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.PosteriorVisualizer), 
    Target = typeof(Bonsai.ML.PointProcessDecoder.GetDecoderData))]
[assembly: TypeVisualizer(typeof(Bonsai.ML.PointProcessDecoder.Design.PosteriorVisualizer), 
    Target = typeof(Bonsai.ML.PointProcessDecoder.GetClassifierData))]

namespace Bonsai.ML.PointProcessDecoder.Design
{
    public class PosteriorVisualizer : MashupVisualizer, IDecoderVisualizer
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
        private double[] _stateSpaceMin;
        private double[] _stateSpaceMax;
        private string _modelName;
        private bool _success = false;
        private Tensor _dataTensor;
        private Func<object, Tensor> _convertInputData; 
        
        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            IManagedPointProcessModelNode node = null;
            var expressionBuilderGraph = (ExpressionBuilderGraph)provider.GetService(typeof(ExpressionBuilderGraph));
            var typeVisualizerContext = (ITypeVisualizerContext)provider.GetService(typeof(ITypeVisualizerContext));
            if (expressionBuilderGraph != null && typeVisualizerContext != null)
            {
                node = ExpressionBuilder.GetWorkflowElement(
                    expressionBuilderGraph.Where(node => node.Value == typeVisualizerContext.Source)
                        .FirstOrDefault().Value) as IManagedPointProcessModelNode;
            }

            if (node == null)
            {
                throw new InvalidOperationException("The decode node is invalid.");
            }

            _convertInputData = node switch
            {
                Decode _ => input => (Tensor)input,
                GetDecoderData _ => input => ((DecoderData)input).Posterior,
                GetClassifierData _ => input => ((ClassifierData)input).DecoderData.Posterior,
                _ => throw new InvalidOperationException("The node is invalid.")
            };

            _modelName = node.Name;

            _visualizer = new MultidimensionalArrayVisualizer()
            {
                PaletteSelectedIndex = 1,
                RenderMethodSelectedIndex = 0
            };
            
            _visualizer.Load(provider);

            _visualizer.Plot.ValueMin = ValueMin;
            _visualizer.Plot.ValueMax = ValueMax;

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

        private void UpdateModel()
        {
            PointProcessModel model;

            try
            {
                model = PointProcessModelManager.GetModel(_modelName);
            } catch {
                _success = false;
                return;
            }

            if (model == null)
            {
                _success = false;
                return;
            }

            _stateSpaceMin = [.. model.StateSpace.Points
                .min(dim: 0)
                .values
                .to_type(ScalarType.Float64)
                .data<double>()
            ];

            _stateSpaceMax = [.. model.StateSpace.Points
                .max(dim: 0)
                .values
                .to_type(ScalarType.Float64)
                .data<double>()
            ];

            _success = true;
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            var posterior = _convertInputData(value);

            if (posterior.NumberOfElements == 0 || !_success)
            {
                return;
            }

            posterior = posterior.mean([0]);
            
            double xMin;
            double xMax;
            double yMin;
            double yMax;

            if (posterior.Dimensions == 1)
            {

                if (_data == null)
                {
                    _dataTensor = zeros(_capacity, posterior.size(0), dtype: ScalarType.Float64, device: posterior.device);
                }

                _dataTensor = _dataTensor[TensorIndex.Slice(1)];
                _dataTensor = concat([_dataTensor,
                    posterior.to_type(ScalarType.Float64)
                        .unsqueeze(0)
                ], dim: 0);

                _data = (double[,])_dataTensor
                    .data<double>()
                    .ToNDArray();

                xMin = 0;
                xMax = _capacity;
                yMin = _stateSpaceMin[0];
                yMax = _stateSpaceMax[0];
            }
            else
            {

                while (posterior.Dimensions > 2)
                {
                    posterior = posterior.sum(dim: 0);
                }

                _data = (double[,])posterior
                    .to_type(ScalarType.Float64)
                    .data<double>()
                    .ToNDArray();

                xMin = _stateSpaceMin[_stateSpaceMin.Length - 2];
                xMax = _stateSpaceMax[_stateSpaceMax.Length - 2];
                yMin = _stateSpaceMin[_stateSpaceMin.Length - 1];
                yMax = _stateSpaceMax[_stateSpaceMax.Length - 1];
            }

            _visualizer.Plot.UpdateHeatMapSeries(
                xMin, xMax, yMin, yMax, _data
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

            var mergedSource = source.SelectMany(xs => 
                xs.Buffer(timer)
                    .Where(buffer => buffer.Count > 0)
                    .Do(buffer => {
                        if (!_success)
                        {
                            UpdateModel();
                        }
                        ValueMin = _visualizer.Plot.ValueMin;
                        ValueMax = _visualizer.Plot.ValueMax;
                        Show(buffer.LastOrDefault());
                    }));

            var mashupSourceStreams = Observable.Merge(
                MashupSources.Select(mashupSource => {
                    var color = colorCycler.Next();
                    var visualizer = mashupSource.Visualizer as Point2DOverlay;
                    visualizer.Color = color;
                    return mashupSource.Source.Output.SelectMany(xs => 
                        xs.Buffer(timer)
                            .Where(buffer => buffer.Count > 0)
                            .Do(buffer => visualizer.Show(buffer.LastOrDefault()))
                    );
                })
            );

            return Observable.Merge(mergedSource, mashupSourceStreams);
        }
    }
}