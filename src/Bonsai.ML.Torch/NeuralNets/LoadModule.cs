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
        public nn.Module Module { get; set; }

        /// <summary>
        /// The path to the modules state.
        /// </summary>
        [Description("The path to the modules state.")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
        public string ModulePath { get; set; }

        /// <summary>
        /// Loads the modules state from the specified file path.
        /// </summary>
        /// <returns></returns>
        public IObservable<nn.Module> Process()
        {
            return Observable.Return(Module.load(ModulePath));
        }

        /// <summary>
        /// Loads the module's state from the specified file path.
        /// </summary>
        /// <returns></returns>
        public IObservable<nn.Module> Process(IObservable<nn.Module> source)
        {
            return source.Select(module =>
            {
                return module.load(ModulePath);
            });
        }
    }
}