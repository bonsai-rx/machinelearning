using System;
using System.ComponentModel;
using System.Reactive.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets
{
    /// <summary>
    /// Loads a TorchScript module from the specified file path.
    /// </summary>
    [Combinator]
    [ResetCombinator]
    [Description("Loads a TorchScript module from the specified file path.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class LoadScriptModule
    {
        /// <summary>
        /// The device on which to load the model.
        /// </summary>
        [Description("The device on which to load the model.")]
        [XmlIgnore]
        public Device Device { get; set; }

        /// <summary>
        /// The path to the TorchScript model file.
        /// </summary>
        [Description("The path to the TorchScript model file.")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
        public string ModelPath { get; set; }

        /// <summary>
        /// Loads the TorchScript module from the specified file path.
        /// </summary>
        /// <returns></returns>
        public IObservable<IModule<Tensor, Tensor>> Process()
        {
            var scriptModule = Device is null ? jit.load<Tensor, Tensor>(ModelPath) : jit.load<Tensor, Tensor>(ModelPath, Device);
            return Observable.Return(scriptModule);
        }
    }
}