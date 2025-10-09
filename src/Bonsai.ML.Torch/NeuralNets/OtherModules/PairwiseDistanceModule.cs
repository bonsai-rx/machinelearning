using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.OtherModules;

/// <summary>
/// Creates a PairwiseDistance module.
/// </summary>
[Combinator]
[Description("Creates a PairwiseDistance module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PairwiseDistanceModule
{
    /// <summary>
    /// The p parameter for the PairwiseDistance module.
    /// </summary>
    [Description("The p parameter for the PairwiseDistance module")]
    public double P { get; set; } = 2D;

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-06D;

    /// <summary>
    /// The keep_dim parameter for the PairwiseDistance module.
    /// </summary>
    [Description("The keep_dim parameter for the PairwiseDistance module")]
    public bool KeepDim { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a PairwiseDistanceModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(PairwiseDistance(P, Eps, KeepDim));
    }
}
