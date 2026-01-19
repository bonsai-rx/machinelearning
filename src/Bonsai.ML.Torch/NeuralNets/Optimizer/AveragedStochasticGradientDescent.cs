using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets.Optimizer;

/// <summary>
/// Represents an operator that creates an averaged stochastic gradient descent (ASGD) optimizer.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.ASGD.html"/> for more information.
/// </remarks>
[Description("Creates an averaged stochastic gradient descent (ASGD) optimizer.")]
[DisplayName("ASGD")]
public class AveragedStochasticGradientDescent
{
    /// <summary>
    /// The learning rate.
    /// </summary>
    [Description("The learning rate.")]
    public double LearningRate { get; set; } = 0.001D;

    /// <summary>
    /// The lambda parameter, which determines the decay term.
    /// </summary>
    [Description("The lambda parameter, which determines the decay term.")]
    public double Lambda { get; set; } = 0.0001D;

    /// <summary>
    /// The alpha parameter, which controls the power law decay.
    /// </summary>
    [Description("The alpha parameter, which controls the power law decay.")]
    public double Alpha { get; set; } = 0.75D;

    /// <summary>
    /// The point when averaging starts.
    /// </summary>
    [Description("The point when averaging starts.")]
    public double T0 { get; set; } = 1000000D;

    /// <summary>
    /// The weight decay coefficient, which adds L2 regularization to the loss.
    /// </summary>
    [Description("The weight decay coefficient, which adds L2 regularization to the loss.")]
    public double WeightDecay { get; set; } = 0D;

    /// <summary>
    /// If set to true, performs maximization instead of minimization of the params based on the objective.
    /// </summary>
    [Description("If set to true, performs maximization instead of minimization of the params based on the objective.")]
    public bool Maximize { get; set; } = false;

    /// <summary>
    /// Creates an ASGD optimizer from the input parameter collection.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.ASGD> Process<T>(IObservable<T> source) where T : IEnumerable<Parameter>
    {
        return source.Select(parameters => ASGD(parameters, LearningRate, Lambda, Alpha, T0, WeightDecay, Maximize));
    }
}
