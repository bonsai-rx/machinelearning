using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using System.Xml.Serialization;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.jit;
using Bonsai.Expressions;

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
        public string? ModelPath { get; set; }

        public IObservable<ScriptModule> Process()
        {
            return Observable.Return(load(ModelPath, Device));
        }
    }
}