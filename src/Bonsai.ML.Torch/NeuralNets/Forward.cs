using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using System.Xml.Serialization;
using static TorchSharp.torch.nn;
using Bonsai.Expressions;

namespace Bonsai.ML.Torch.NeuralNets
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Forward
    {
        [XmlIgnore]
        public Module Model { get; set; }

        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            var model = (Module<Tensor, Tensor>) Model;
            return source.Select(model.forward);
        }
    }
}