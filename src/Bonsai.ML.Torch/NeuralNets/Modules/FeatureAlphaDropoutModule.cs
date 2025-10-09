using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Modules;

/// <summary>
/// Creates a FeatureAlphaDropout module.
/// </summary>
[Combinator]
[Description("Creates a FeatureAlphaDropout module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class FeatureAlphaDropoutModule
{
    /// <summary>
    /// The p parameter for the FeatureAlphaDropout module.
    /// </summary>
    [Description("The p parameter for the FeatureAlphaDropout module")]
    public double P { get; set; } = 0.5D;

    /// <summary>
    /// Generates an observable sequence that creates a FeatureAlphaDropoutModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(FeatureAlphaDropout(P));
    }
}
