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
/// Creates a Adamax optimizer module.
/// </summary>
[Combinator]
[Description("Creates a Adamax optimizer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class AdamaxOptimizerModule
{
    /// <summary>
    /// The parameters parameter for the Adamax module.
    /// </summary>
    [Description("The parameters parameter for the Adamax module")]
    public IEnumerable<Parameter> Parameters { get; set; }

    /// <summary>
    /// The lr parameter for the Adamax module.
    /// </summary>
    [Description("The lr parameter for the Adamax module")]
    public double Lr { get; set; } = 0.002;

    /// <summary>
    /// The beta1 parameter for the Adamax module.
    /// </summary>
    [Description("The beta1 parameter for the Adamax module")]
    public double Beta1 { get; set; } = 0.9;

    /// <summary>
    /// The beta2 parameter for the Adamax module.
    /// </summary>
    [Description("The beta2 parameter for the Adamax module")]
    public double Beta2 { get; set; } = 0.999;

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-08;

    /// <summary>
    /// The weight_decay parameter for the Adamax module.
    /// </summary>
    [Description("The weight_decay parameter for the Adamax module")]
    public double WeightDecay { get; set; } = 0;

    /// <summary>
    /// Generates an observable sequence that creates a AdamaxOptimizer module.
    /// </summary>
    public IObservable<Optimizer> Process()
    {
        return Observable.Return(Adamax(Parameters, Lr, Beta1, Beta2, Eps, WeightDecay));
    }
}
