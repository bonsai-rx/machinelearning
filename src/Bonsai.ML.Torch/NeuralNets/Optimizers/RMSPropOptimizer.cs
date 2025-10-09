using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets.Optimizers;

/// <summary>
/// Creates a RMSProp optimizer.
/// </summary>
[Combinator]
[Description("Creates a RMSProp optimizer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class RMSPropOptimizer
{
    /// <summary>
    /// The parameters parameter for the RMSProp module.
    /// </summary>
    [Description("The parameters parameter for the RMSProp module")]
    public IEnumerable<Parameter> Parameters { get; set; }

    /// <summary>
    /// The lr parameter for the RMSProp module.
    /// </summary>
    [Description("The lr parameter for the RMSProp module")]
    public double Lr { get; set; } = 0.01D;

    /// <summary>
    /// The alpha parameter for the RMSProp module.
    /// </summary>
    [Description("The alpha parameter for the RMSProp module")]
    public double Alpha { get; set; } = 0.99D;

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-08D;

    /// <summary>
    /// The weight_decay parameter for the RMSProp module.
    /// </summary>
    [Description("The weight_decay parameter for the RMSProp module")]
    public double WeightDecay { get; set; } = 0D;

    /// <summary>
    /// The value used for the running_mean and running_var computation.
    /// </summary>
    [Description("The value used for the running_mean and running_var computation")]
    public double Momentum { get; set; } = 0D;

    /// <summary>
    /// The centered parameter for the RMSProp module.
    /// </summary>
    [Description("The centered parameter for the RMSProp module")]
    public bool Centered { get; set; } = false;

    /// <summary>
    /// The maximize parameter for the RMSProp module.
    /// </summary>
    [Description("The maximize parameter for the RMSProp module")]
    public bool Maximize { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a RMSPropOptimizer.
    /// </summary>
    public IObservable<Optimizer> Process()
    {
        return Observable.Return(RMSProp(Parameters, Lr, Alpha, Eps, WeightDecay, Momentum, Centered, Maximize));
    }
}
