using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using System.Xml.Serialization;
using TorchSharp.Modules;

namespace Bonsai.ML.Torch.NeuralNets
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Forward
    {
        [XmlIgnore]
        public ITorchModule Model { get; set; }

        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(Model.forward);
        }
    }
}