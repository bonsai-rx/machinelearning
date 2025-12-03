using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Saves the module to a file.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Saves the module to a file.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class LoadModule
{
    /// <summary>
    /// The path to the modules state.
    /// </summary>
    [Description("The path to the modules state.")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string ModulePath { get; set; }

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