using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Modules;

/// <summary>
/// Creates a LeakyReLU activation function module.
/// </summary>
[Combinator]
[Description("Creates a LeakyReLU activation function module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LeakyReLUModule
{
    /// <summary>
    /// The negative_slope parameter for the LeakyReLU module.
    /// </summary>
    [Description("The negative_slope parameter for the LeakyReLU module")]
    public double NegativeSlope { get; set; } = 0.01;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a LeakyReLU module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(LeakyReLU(NegativeSlope, Inplace));
    }
}
