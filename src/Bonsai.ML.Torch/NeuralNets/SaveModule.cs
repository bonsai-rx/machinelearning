using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets;

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
    /// The path to save the module.
    /// </summary>
    [Description("The path to save the module.")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string ModulePath { get; set; }

    /// <summary>
    /// Saves the input module to the specified file path.
    /// </summary>
    public IObservable<Module<Tensor, Tensor>> Process(IObservable<Module<Tensor, Tensor>> source)
    {
        return source.Do(input =>
        {
            input.save(ModulePath);
        });
    }
}