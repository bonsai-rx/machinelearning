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
/// Creates a Rprop optimizer module.
/// </summary>
[Combinator]
[Description("Creates a Rprop optimizer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class RpropOptimizerModule
{
    /// <summary>
    /// The parameters parameter for the Rprop module.
    /// </summary>
    [Description("The parameters parameter for the Rprop module")]
    public IEnumerable<Parameter> Parameters { get; set; }

    /// <summary>
    /// The lr parameter for the Rprop module.
    /// </summary>
    [Description("The lr parameter for the Rprop module")]
    public double Lr { get; set; } = 0.01;

    /// <summary>
    /// The etaminus parameter for the Rprop module.
    /// </summary>
    [Description("The etaminus parameter for the Rprop module")]
    public double Etaminus { get; set; } = 0.5;

    /// <summary>
    /// The etaplus parameter for the Rprop module.
    /// </summary>
    [Description("The etaplus parameter for the Rprop module")]
    public double Etaplus { get; set; } = 1.2;

    /// <summary>
    /// The min_step parameter for the Rprop module.
    /// </summary>
    [Description("The min_step parameter for the Rprop module")]
    public double MinStep { get; set; } = 1E-06;

    /// <summary>
    /// The max_step parameter for the Rprop module.
    /// </summary>
    [Description("The max_step parameter for the Rprop module")]
    public double MaxStep { get; set; } = 50;

    /// <summary>
    /// The maximize parameter for the Rprop module.
    /// </summary>
    [Description("The maximize parameter for the Rprop module")]
    public bool Maximize { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a RpropOptimizer module.
    /// </summary>
    public IObservable<Optimizer> Process()
    {
        return Observable.Return(Rprop(Parameters, Lr, Etaminus, Etaplus, MinStep, MaxStep, Maximize));
    }
}
