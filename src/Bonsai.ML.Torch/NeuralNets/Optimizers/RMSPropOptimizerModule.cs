using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Collections.Generic;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets.Optimizers;

/// <summary>
/// Creates a RMSProp optimizer module.
/// </summary>
[Combinator]
[Description("Creates a RMSProp optimizer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class RMSPropOptimizerModule
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
    public double Lr { get; set; } = 0.01;

    /// <summary>
    /// The alpha parameter for the RMSProp module.
    /// </summary>
    [Description("The alpha parameter for the RMSProp module")]
    public double Alpha { get; set; } = 0.99;

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-08;

    /// <summary>
    /// The weight_decay parameter for the RMSProp module.
    /// </summary>
    [Description("The weight_decay parameter for the RMSProp module")]
    public double WeightDecay { get; set; } = 0;

    /// <summary>
    /// The value used for the running_mean and running_var computation.
    /// </summary>
    [Description("The value used for the running_mean and running_var computation")]
    public double Momentum { get; set; } = 0;

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
    /// Generates an observable sequence that creates a RMSPropOptimizer module.
    /// </summary>
    public IObservable<Optimizer> Process()
    {
        return Observable.Return(RMSProp(Parameters, Lr, Alpha, Eps, WeightDecay, Momentum, Centered, Maximize));
    }
}
