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
/// Creates a ASGD optimizer module.
/// </summary>
[Combinator]
[Description("Creates a ASGD optimizer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ASGDOptimizerModule
{
    /// <summary>
    /// The parameters parameter for the ASGD module.
    /// </summary>
    [Description("The parameters parameter for the ASGD module")]
    public IEnumerable<Parameter> Parameters { get; set; }

    /// <summary>
    /// The lr parameter for the ASGD module.
    /// </summary>
    [Description("The lr parameter for the ASGD module")]
    public double Lr { get; set; } = 0.001;

    /// <summary>
    /// The lambd parameter for the ASGD module.
    /// </summary>
    [Description("The lambd parameter for the ASGD module")]
    public double Lambd { get; set; } = 0.0001;

    /// <summary>
    /// The alpha parameter for the ASGD module.
    /// </summary>
    [Description("The alpha parameter for the ASGD module")]
    public double Alpha { get; set; } = 0.75;

    /// <summary>
    /// The t0 parameter for the ASGD module.
    /// </summary>
    [Description("The t0 parameter for the ASGD module")]
    public double T0 { get; set; } = 1000000;

    /// <summary>
    /// The weight_decay parameter for the ASGD module.
    /// </summary>
    [Description("The weight_decay parameter for the ASGD module")]
    public double WeightDecay { get; set; } = 0;

    /// <summary>
    /// The maximize parameter for the ASGD module.
    /// </summary>
    [Description("The maximize parameter for the ASGD module")]
    public bool Maximize { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a ASGDOptimizer module.
    /// </summary>
    public IObservable<Optimizer> Process()
    {
        return Observable.Return(ASGD(Parameters, Lr, Lambd, Alpha, T0, WeightDecay, Maximize));
    }
}
