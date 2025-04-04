using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
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
        public IObservable<ITorchModule> Process()
        {
            var scriptModule = jit.load<Tensor, Tensor>(ModelPath, Device);
            var torchModule = new TorchModuleAdapter(scriptModule);
            return Observable.Return((ITorchModule)torchModule);
        }
    }
}