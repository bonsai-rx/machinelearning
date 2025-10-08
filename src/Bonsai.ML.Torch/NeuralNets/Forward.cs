using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
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
            Model.Module.eval();
            return source.Select(Model.Forward);
        }
    }
}