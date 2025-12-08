using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets.Optimizer;

/// <summary>
/// Represents an operator that creates an Adagrad optimizer.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html"/> for more information.
/// </remarks>
[Description("Creates an Adagrad optimizer.")]
public class Adagrad
{
    /// <summary>
    /// The learning rate coefficient.
    /// </summary>
    [Description("The learning rate coefficient.")]
    public double LearningRate { get; set; } = 0.01D;

    /// <summary>
    /// The learning rate decay.
    /// </summary>
    [Description("The learning rate decay.")]
    public double LearningRateDecay { get; set; } = 0D;

    /// <summary>
    /// The weight decay coefficient, which adds L2 regularization to the loss.
    /// </summary>
    [Description("The weight decay coefficient, which adds L2 regularization to the loss.")]
    public double WeightDecay { get; set; } = 0D;

    /// <summary>
    /// The initial value of the sum of squares of gradients.
    /// </summary>
    [Description("The initial value of the sum of squares of gradients.")]
    public double InitialAccumulatorValue { get; set; } = 0D;

    /// <summary>
    /// The value added to the denominator for numerical stability.
    /// </summary>
    [Description("The value added to the denominator for numerical stability.")]
    public double Eps { get; set; } = 1E-10D;

    /// <summary>
    /// Creates an Adagrad optimizer from the input parameter collection.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<optim.Optimizer> Process<T>(IObservable<T> source) where T : IEnumerable<Parameter>
    {
        return source.Select(parameters => Adagrad(parameters, LearningRate, LearningRateDecay, WeightDecay, InitialAccumulatorValue, Eps));
    }
}
