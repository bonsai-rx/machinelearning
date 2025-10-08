using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Losses;

/// <summary>
/// Creates a HuberLoss module module.
/// </summary>
[Combinator]
[Description("Creates a HuberLoss module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class HuberLossModule
{
    /// <summary>
    /// The delta parameter for the HuberLoss module.
    /// </summary>
    [Description("The delta parameter for the HuberLoss module")]
    public double Delta { get; set; } = 1;

    /// <summary>
    /// The reduction parameter for the HuberLoss module.
    /// </summary>
    [Description("The reduction parameter for the HuberLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a HuberLoss module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(HuberLoss(Delta, Reduction));
    }
}
