using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class LoadScriptModule
    {

        [XmlIgnore]
        public Device Device { get; set; } = CPU;

        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
        public string ModelPath { get; set; }

        public IObservable<ITorchModule> Process()
        {
            return Observable.Return((ITorchModule)jit.load<Tensor, Tensor>(ModelPath, Device));
        }
    }
}