using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Collections.Generic;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets.Optimizers;

/// <summary>
/// Creates a Adadelta optimizer module.
/// </summary>
[Combinator]
[Description("Creates a Adadelta optimizer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class AdadeltaOptimizerModule
{
    /// <summary>
    /// The parameters parameter for the Adadelta module.
    /// </summary>
    [Description("The parameters parameter for the Adadelta module")]
    public IEnumerable<Parameter> Parameters { get; set; }

    /// <summary>
    /// The lr parameter for the Adadelta module.
    /// </summary>
    [Description("The lr parameter for the Adadelta module")]
    public double Lr { get; set; } = 1;

    /// <summary>
    /// The rho parameter for the Adadelta module.
    /// </summary>
    [Description("The rho parameter for the Adadelta module")]
    public double Rho { get; set; } = 0.9;

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-06;

    /// <summary>
    /// The weight_decay parameter for the Adadelta module.
    /// </summary>
    [Description("The weight_decay parameter for the Adadelta module")]
    public double WeightDecay { get; set; } = 0;

    /// <summary>
    /// The maximize parameter for the Adadelta module.
    /// </summary>
    [Description("The maximize parameter for the Adadelta module")]
    public bool Maximize { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a AdadeltaOptimizer module.
    /// </summary>
    public IObservable<Optimizer> Process()
    {
        return Observable.Return(Adadelta(Parameters, Lr, Rho, Eps, WeightDecay, Maximize));
    }
}
