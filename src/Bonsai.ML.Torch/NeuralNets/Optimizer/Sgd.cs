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
/// Creates a SGD optimizer.
/// </summary>
[Combinator]
[Description("Creates a SGD optimizer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Sgd
{
    /// <summary>
    /// The parameters parameter for the SGD module.
    /// </summary>
    [Description("The parameters parameter for the SGD module")]
    public IEnumerable<Parameter> Parameters { get; set; }

    /// <summary>
    /// The learningrate parameter for the SGD module.
    /// </summary>
    [Description("The learningrate parameter for the SGD module")]
    public double LearningRate { get; set; }

    /// <summary>
    /// The value used for the running_mean and running_var computation.
    /// </summary>
    [Description("The value used for the running_mean and running_var computation")]
    public double Momentum { get; set; } = 0D;

    /// <summary>
    /// The dampening parameter for the SGD module.
    /// </summary>
    [Description("The dampening parameter for the SGD module")]
    public double Dampening { get; set; } = 0D;

    /// <summary>
    /// The weight_decay parameter for the SGD module.
    /// </summary>
    [Description("The weight_decay parameter for the SGD module")]
    public double WeightDecay { get; set; } = 0D;

    /// <summary>
    /// The nesterov parameter for the SGD module.
    /// </summary>
    [Description("The nesterov parameter for the SGD module")]
    public bool Nesterov { get; set; } = false;

    /// <summary>
    /// The maximize parameter for the SGD module.
    /// </summary>
    [Description("The maximize parameter for the SGD module")]
    public bool Maximize { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a SGDOptimizer.
    /// </summary>
    public IObservable<optim.Optimizer> Process()
    {
        return Observable.Return(SGD(Parameters, LearningRate, Momentum, Dampening, WeightDecay, Nesterov, Maximize));
    }
}
