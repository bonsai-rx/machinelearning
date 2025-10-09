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
/// Creates a 2D dropout regularization layer.
/// </summary>
[Combinator]
[Description("Creates a 2D dropout regularization layer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Dropout2dModule
{
    /// <summary>
    /// The p parameter for the Dropout2d module.
    /// </summary>
    [Description("The p parameter for the Dropout2d module")]
    public double P { get; set; } = 0.5D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a Dropout2dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Dropout2d(P, Inplace));
    }
}
