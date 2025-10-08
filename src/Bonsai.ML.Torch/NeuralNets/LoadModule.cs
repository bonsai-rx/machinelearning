using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets
{
    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    [Combinator]
    [ResetCombinator]
    [Description("Saves the model to a file.")]
    [WorkflowElementCategory(ElementCategory.Sink)]
    public class LoadModule
    {
        /// <summary>
        /// The model to save.
        /// </summary>
        [Description("The model to save.")]
        [XmlIgnore]
        public Module Module { get; set; }

        /// <summary>
        /// The path to save the model.
        /// </summary>
        [Description("The path to save the model.")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
        public string ModelPath { get; set; }

        /// <summary>
        /// Saves the model to the specified file path.
        /// </summary>
        /// <returns></returns>
        public IObservable<Module> Process()
        {
            return Observable.Return(Module.load(ModelPath));
        }
    }
}