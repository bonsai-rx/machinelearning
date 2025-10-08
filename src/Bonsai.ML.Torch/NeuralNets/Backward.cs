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
        public IModule<Tensor, Tensor> Model { get; set; }

        /// <summary>
        /// The loss function to use for training.
        /// </summary>
        public IModule<Tensor, Tensor, Tensor> Loss { get; set; }

        /// <summary>
        /// Trains the model using backpropagation.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor>> source)
        {
            var model = Model as Module<Tensor, Tensor>;
            var loss = Loss as Module<Tensor, Tensor, Tensor>;
            var scheduler = lr_scheduler.StepLR(Optimizer, 1, 0.7);
            model.train();

            return source.Select((input) => {
                var (data, target) = input;
                using (_ = NewDisposeScope())
                {
                    Optimizer.zero_grad();
                    
                    var prediction = model.forward(data);
                    var output = loss.forward(prediction, target);

                    output.backward();

                    Optimizer.step();
                    return output.MoveToOuterDisposeScope();
                }
            });
        }
    }
}