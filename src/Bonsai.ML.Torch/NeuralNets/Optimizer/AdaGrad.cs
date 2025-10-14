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
/// Creates a Adagrad optimizer.
/// </summary>
[Combinator]
[Description("Creates a Adagrad optimizer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class AdaGrad
{
    /// <summary>
    /// The parameters parameter for the Adagrad module.
    /// </summary>
    [Description("The parameters parameter for the Adagrad module")]
    public IEnumerable<Parameter> Parameters { get; set; }

    /// <summary>
    /// The lr parameter for the Adagrad module.
    /// </summary>
    [Description("The lr parameter for the Adagrad module")]
    public double Lr { get; set; } = 0.01D;

    /// <summary>
    /// The lr_decay parameter for the Adagrad module.
    /// </summary>
    [Description("The lr_decay parameter for the Adagrad module")]
    public double LrDecay { get; set; } = 0D;

    /// <summary>
    /// The weight_decay parameter for the Adagrad module.
    /// </summary>
    [Description("The weight_decay parameter for the Adagrad module")]
    public double WeightDecay { get; set; } = 0D;

    /// <summary>
    /// The initial_accumulator_value parameter for the Adagrad module.
    /// </summary>
    [Description("The initial_accumulator_value parameter for the Adagrad module")]
    public double InitialAccumulatorValue { get; set; } = 0D;

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-10D;

    /// <summary>
    /// Generates an observable sequence that creates a AdagradOptimizer.
    /// </summary>
    public IObservable<optim.Optimizer> Process()
    {
        return Observable.Return(Adagrad(Parameters, Lr, LrDecay, WeightDecay, InitialAccumulatorValue, Eps));
    }
}
