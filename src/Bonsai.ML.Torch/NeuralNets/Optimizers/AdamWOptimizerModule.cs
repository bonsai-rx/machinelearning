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
/// Creates a AdamW optimizer module.
/// </summary>
[Combinator]
[Description("Creates a AdamW optimizer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class AdamWOptimizerModule
{
    /// <summary>
    /// The parameters parameter for the AdamW module.
    /// </summary>
    [Description("The parameters parameter for the AdamW module")]
    public IEnumerable<Parameter> Parameters { get; set; }

    /// <summary>
    /// The lr parameter for the AdamW module.
    /// </summary>
    [Description("The lr parameter for the AdamW module")]
    public double Lr { get; set; } = 0.001;

    /// <summary>
    /// The beta1 parameter for the AdamW module.
    /// </summary>
    [Description("The beta1 parameter for the AdamW module")]
    public double Beta1 { get; set; } = 0.9;

    /// <summary>
    /// The beta2 parameter for the AdamW module.
    /// </summary>
    [Description("The beta2 parameter for the AdamW module")]
    public double Beta2 { get; set; } = 0.999;

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-08;

    /// <summary>
    /// The weight_decay parameter for the AdamW module.
    /// </summary>
    [Description("The weight_decay parameter for the AdamW module")]
    public double WeightDecay { get; set; } = 0;

    /// <summary>
    /// The amsgrad parameter for the AdamW module.
    /// </summary>
    [Description("The amsgrad parameter for the AdamW module")]
    public bool Amsgrad { get; set; } = false;

    /// <summary>
    /// The maximize parameter for the AdamW module.
    /// </summary>
    [Description("The maximize parameter for the AdamW module")]
    public bool Maximize { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a AdamWOptimizer module.
    /// </summary>
    public IObservable<Optimizer> Process()
    {
        return Observable.Return(AdamW(Parameters, Lr, Beta1, Beta2, Eps, WeightDecay, Amsgrad, Maximize));
    }
}
