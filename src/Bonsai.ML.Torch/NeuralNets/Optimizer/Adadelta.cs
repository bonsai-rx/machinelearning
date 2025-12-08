using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets.Optimizer;

/// <summary>
/// Represents an operator that creates an Adadelta optimizer.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html"/> for more information.
/// </remarks>
[Description("Creates an Adadelta optimizer.")]
public class Adadelta
{
    /// <summary>
    /// The learning rate coefficient that scales delta before it is applied to the parameters.
    /// </summary>
    [Description("The learning rate coefficient that scales delta before it is applied to the parameters.")]
    public double LearningRate { get; set; } = 1D;

    /// <summary>
    /// The coefficient used to calculate a running average of squared gradients.
    /// </summary>
    [Description("The coefficient used to calculate a running average of squared gradients.")]
    public double Rho { get; set; } = 0.9D;

    /// <summary>
    /// The value added to the denominator for numerical stability.
    /// </summary>
    [Description("The value added to the denominator for numerical stability.")]
    public double Eps { get; set; } = 1E-06D;

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
    /// Creates an Adadelta optimizer from the input parameter collection.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<optim.Optimizer> Process<T>(IObservable<T> source) where T : IEnumerable<Parameter>
    {
        return source.Select(parameters => Adadelta(parameters, LearningRate, Rho, Eps, WeightDecay, Maximize));
    }
}
