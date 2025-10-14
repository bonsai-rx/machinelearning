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

namespace Bonsai.ML.Torch.NeuralNets.Optimizer;

/// <summary>
/// Creates a AdamW optimizer.
/// </summary>
[Combinator]
[Description("Creates a AdamW optimizer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class AdamW
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
    public double Lr { get; set; } = 0.001D;

    /// <summary>
    /// The beta1 parameter for the AdamW module.
    /// </summary>
    [Description("The beta1 parameter for the AdamW module")]
    public double Beta1 { get; set; } = 0.9D;

    /// <summary>
    /// The beta2 parameter for the AdamW module.
    /// </summary>
    [Description("The beta2 parameter for the AdamW module")]
    public double Beta2 { get; set; } = 0.999D;

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-08D;

    /// <summary>
    /// The weight_decay parameter for the AdamW module.
    /// </summary>
    [Description("The weight_decay parameter for the AdamW module")]
    public double WeightDecay { get; set; } = 0D;

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
    /// Generates an observable sequence that creates a AdamWOptimizer.
    /// </summary>
    public IObservable<optim.Optimizer> Process()
    {
        return Observable.Return(AdamW(Parameters, Lr, Beta1, Beta2, Eps, WeightDecay, Amsgrad, Maximize));
    }
}
