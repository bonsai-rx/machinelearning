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
/// Creates a GaussianNLLLoss module module.
/// </summary>
[Combinator]
[Description("Creates a GaussianNLLLoss module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class GaussianNLLLossModule
{
    /// <summary>
    /// The full parameter for the GaussianNLLLoss module.
    /// </summary>
    [Description("The full parameter for the GaussianNLLLoss module")]
    public bool Full { get; set; } = false;

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public float Eps { get; set; } = 1E-08F;

    /// <summary>
    /// The reduction parameter for the GaussianNLLLoss module.
    /// </summary>
    [Description("The reduction parameter for the GaussianNLLLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a GaussianNLLLoss module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(GaussianNLLLoss(Full, Eps, Reduction));
    }
}
