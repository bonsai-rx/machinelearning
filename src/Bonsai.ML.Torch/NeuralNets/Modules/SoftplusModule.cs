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
/// Creates a Softplus module.
/// </summary>
[Combinator]
[Description("Creates a Softplus module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SoftplusModule
{
    /// <summary>
    /// The beta parameter for the Softplus module.
    /// </summary>
    [Description("The beta parameter for the Softplus module")]
    public double Beta { get; set; } = 1D;

    /// <summary>
    /// The threshold parameter for the Softplus module.
    /// </summary>
    [Description("The threshold parameter for the Softplus module")]
    public double Threshold { get; set; } = 20D;

    /// <summary>
    /// Generates an observable sequence that creates a SoftplusModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Softplus(Beta, Threshold));
    }
}
