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
    public class SaveModule
    {
        /// <summary>
        /// The module to save.
        /// </summary>
        [Description("The module to save.")]
        [XmlIgnore]
        public IModule<Tensor, Tensor> Module { get; set; }

        /// <summary>
        /// The path to save the module.
        /// </summary>
        [Description("The path to save the module.")]
        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
        public string ModulePath { get; set; }

        /// <summary>
        /// Saves the input module to the specified file path.
        /// </summary>
        public IObservable<IModule<Tensor, Tensor>> Process(IObservable<IModule<Tensor, Tensor>> source)
        {
            return source.Do(input =>
            {
                var module = input as nn.Module;
                module?.save(ModulePath);
            });
        }

        /// <summary>
        /// Saves the module to the specified file path.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<T> Process<T>(IObservable<T> source)
        {
            return source.Do(input =>
            {
                var module = Module as nn.Module;
                module?.save(ModulePath);
            });
        }
    }
}