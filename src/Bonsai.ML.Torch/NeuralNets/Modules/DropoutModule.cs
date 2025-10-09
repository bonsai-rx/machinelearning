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
/// Creates a Dropout regularization layer.
/// </summary>
[Combinator]
[Description("Creates a Dropout regularization layer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class DropoutModule
{
    /// <summary>
    /// The p parameter for the Dropout module.
    /// </summary>
    [Description("The p parameter for the Dropout module")]
    public double P { get; set; } = 0.5D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a DropoutModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Dropout(P, Inplace));
    }
}
