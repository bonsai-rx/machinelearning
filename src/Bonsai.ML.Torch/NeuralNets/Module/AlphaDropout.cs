using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Module;

/// <summary>
/// Creates a AlphaDropout module.
/// </summary>
[Combinator]
[Description("Creates a AlphaDropout module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class AlphaDropout
{
    /// <summary>
    /// The p parameter for the AlphaDropout module.
    /// </summary>
    [Description("The p parameter for the AlphaDropout module")]
    public double P { get; set; } = 0.5D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a AlphaDropoutModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(AlphaDropout(P, Inplace));
    }
}
