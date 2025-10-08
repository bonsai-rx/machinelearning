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
/// Creates a Softplus module module.
/// </summary>
[Combinator]
[Description("Creates a Softplus module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SoftplusModule
{
    /// <summary>
    /// The beta parameter for the Softplus module.
    /// </summary>
    [Description("The beta parameter for the Softplus module")]
    public double Beta { get; set; } = 1;

    /// <summary>
    /// The threshold parameter for the Softplus module.
    /// </summary>
    [Description("The threshold parameter for the Softplus module")]
    public double Threshold { get; set; } = 20;

    /// <summary>
    /// Generates an observable sequence that creates a Softplus module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Softplus(Beta, Threshold));
    }
}
