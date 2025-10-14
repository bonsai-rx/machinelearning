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
/// Creates a dropout layer.
/// </summary>
[Combinator]
[Description("Creates a dropout layer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Dropout
{
    /// <summary>
    /// The number of dimensions for the AdaptiveAvgPool module.
    /// </summary>
    [Description("The number of dimensions for the AdaptiveAvgPool module")]
    public Dimensions Dimensions { get; set; } = Dimensions.None;

    /// <summary>
    /// The p parameter for the Dropout1d module.
    /// </summary>
    [Description("The p parameter for the Dropout1d module")]
    public double P { get; set; } = 0.5D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a Dropout module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Dimensions switch
        {
            Dimensions.None => Observable.Return(Dropout(P, Inplace)),
            Dimensions.One => Observable.Return(Dropout1d(P, Inplace)),
            Dimensions.Two => Observable.Return(Dropout2d(P, Inplace)),
            Dimensions.Three => Observable.Return(Dropout3d(P, Inplace)),
            _ => throw new InvalidOperationException("The specified number of dimensions is not supported."),
        };
    }
}
