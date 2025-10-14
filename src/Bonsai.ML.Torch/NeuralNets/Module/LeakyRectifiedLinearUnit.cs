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
/// Creates a LeakyReLU activation function.
/// </summary>
[Combinator]
[Description("Creates a LeakyReLU activation function.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LeakyRectifiedLinearUnit
{
    /// <summary>
    /// The negative_slope parameter for the LeakyReLU module.
    /// </summary>
    [Description("The negative_slope parameter for the LeakyReLU module")]
    public double NegativeSlope { get; set; } = 0.01D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a LeakyReLUModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(LeakyReLU(NegativeSlope, Inplace));
    }
}
