using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets.Optimizer;

/// <summary>
/// Represents an operator that creates a stochastic gradient descent (SGD) optimizer.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.SGD.html"/> for more information.
/// </remarks>
[Description("Creates a stochastic gradient descent (SGD) optimizer.")]
[DisplayName("SGD")]
public class StochasticGradientDescent
{
    /// <summary>
    /// The learning rate.
    /// </summary>
    [Description("The learning rate.")]
    public double LearningRate { get; set; }

    /// <summary>
    /// The momentum factor.
    /// </summary>
    [Description("The momentum factor.")]
    public double Momentum { get; set; } = 0D;

    /// <summary>
    /// The dampening for momentum.
    /// </summary>
    [Description("The dampening for momentum.")]
    public double Dampening { get; set; } = 0D;

    /// <summary>
    /// The weight decay coefficient, which adds L2 regularization to the loss.
    /// </summary>
    [Description("The weight decay coefficient, which adds L2 regularization to the loss.")]
    public double WeightDecay { get; set; } = 0D;

    /// <summary>
    /// If set to true, enables Nesterov momentum when momentum is non-zero.
    /// </summary>
    [Description("If set to true, enables Nesterov momentum when momentum is non-zero.")]
    public bool Nesterov { get; set; } = false;

    /// <summary>
    /// If set to true, performs maximization instead of minimization of the params based on the objective.
    /// </summary>
    [Description("If set to true, performs maximization instead of minimization of the params based on the objective.")]
    public bool Maximize { get; set; } = false;

    /// <summary>
    /// Creates an SGD optimizer from the input parameter collection.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.SGD> Process<T>(IObservable<T> source) where T : IEnumerable<Parameter>
    {
        return source.Select(parameters => SGD(parameters, LearningRate, Momentum, Dampening, WeightDecay, Nesterov, Maximize));
    }
}
