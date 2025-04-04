using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets
{
    /// <summary>
    /// Trains a model using backpropagation.
    /// </summary>
    [Combinator]
    [ResetCombinator]
    [Description("Trains a model using backpropagation.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Backward
    {
        /// <summary>
        /// The optimizer to use for training.
        /// </summary>
        public Optimizer Optimizer { get; set; }

        /// <summary>
        /// The model to train.
        /// </summary>
        [XmlIgnore]
        public ITorchModule Model { get; set; }

        /// <summary>
        /// The loss function to use for training.
        /// </summary>
        public Loss Loss { get; set; }

        /// <summary>
        /// Trains the model using backpropagation.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor>> source)
        {
            optim.Optimizer optimizer = Optimizer switch
            {
                Optimizer.Adam => Adam(Model.Module.parameters()),
                _ => throw new ArgumentException($"Selected optimizer, {Optimizer} is currently not supported.")
            };

            Module<Tensor, Tensor, Tensor> loss = Loss switch
            {
                Loss.NegativeLogLikelihood => NLLLoss(),
                _ => throw new ArgumentException($"Selected loss, {Loss} is currently not supported.")
            };

            var scheduler = lr_scheduler.StepLR(optimizer, 1, 0.7);
            Model.Module.train();

            return source.Select((input) => {
                var (data, target) = input;
                using (_ = NewDisposeScope())
                {
                    optimizer.zero_grad();
                    
                    var prediction = Model.Forward(data);
                    var output = loss.forward(prediction, target);

                    output.backward();

                    optimizer.step();
                    return output.MoveToOuterDisposeScope();
                }
            });
        }
    }
}