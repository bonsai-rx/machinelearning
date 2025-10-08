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
/// Creates a RReLU module module.
/// </summary>
[Combinator]
[Description("Creates a RReLU module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class RReLUModule
{
    /// <summary>
    /// The lower parameter for the RReLU module.
    /// </summary>
    [Description("The lower parameter for the RReLU module")]
    public double Lower { get; set; } = 0.125;

    /// <summary>
    /// The upper parameter for the RReLU module.
    /// </summary>
    [Description("The upper parameter for the RReLU module")]
    public double Upper { get; set; } = 0.3333333333333333;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a RReLU module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(RReLU(Lower, Upper, Inplace));
    }
}
