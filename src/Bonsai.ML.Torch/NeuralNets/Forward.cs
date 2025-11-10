using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.jit;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets
{
    /// <summary>
    /// Runs forward inference on the input tensor using the specified model.
    /// </summary>
    [Combinator]
    [ResetCombinator]
    [Description("Runs forward inference on the input tensor using the specified model.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Forward
    {
        /// <summary>
        /// The model to use for inference.
        /// </summary>
        [XmlIgnore]
        public IModule<Tensor, Tensor> Model { get; set; }

        /// <summary>
        /// Runs forward inference on the input tensor using the specified model.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(input =>
            {
                return Model switch
                {
                    Module<Tensor, Tensor> module => module.forward(input),
                    ScriptModule<Tensor, Tensor> scriptModule => scriptModule.forward(input),
                    _ => throw new InvalidOperationException("Unsupported model type for forward inference.")
                };
            });
        }
    }
}