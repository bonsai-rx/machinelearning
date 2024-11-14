using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using System.Xml.Serialization;
using TorchSharp.Modules;
using TorchSharp;

namespace Bonsai.ML.Torch.NeuralNets
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Sink)]
    public class SaveModel
    {
        [XmlIgnore]
        public ITorchModule Model { get; set; }

        public string ModelPath { get; set; }

        public IObservable<T> Process<T>(IObservable<T> source)
        {
            return source.Do(input => {
                Model.Module.save(ModelPath);
            });
        }
    }
}