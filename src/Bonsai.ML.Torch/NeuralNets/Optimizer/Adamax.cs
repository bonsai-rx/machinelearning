using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets.Optimizer;

/// <summary>
/// Represents an operator that creates an Adamax optimizer.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html"/> for more information.
/// </remarks>
[Description("Creates an Adamax optimizer.")]
public class Adamax
{
    /// <summary>
    /// The learning rate coefficient.
    /// </summary>
    [Description("The learning rate coefficient.")]
    public double LearningRate { get; set; } = 0.002D;

    /// <summary>
    /// The beta coefficient for computing the running average of the gradient.
    /// </summary>
    [Description("The beta coefficient for computing the running average of the gradient.")]
    public double BetaGradient { get; set; } = 0.9D;

    /// <summary>
    /// The beta coefficient for computing the running average of the squared gradient.
    /// </summary>
    [Description("The beta coefficient for computing the running average of the squared gradient.")]
    public double BetaSquaredGradient { get; set; } = 0.999D;

    /// <summary>
    /// The value added to the denominator for numerical stability.
    /// </summary>
    [Description("The value added to the denominator for numerical stability.")]
    public double Eps { get; set; } = 1E-08D;

    /// <summary>
    /// The weight decay coefficient, which adds L2 regularization to the loss.
    /// </summary>
    [Description("The weight decay coefficient, which adds L2 regularization to the loss.")]
    public double WeightDecay { get; set; } = 0D;

    /// <summary>
    /// Creates an Adamax optimizer from the input parameter collection.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<optim.Optimizer> Process<T>(IObservable<T> source) where T : IEnumerable<Parameter>
    {
        return source.Select(parameters => Adamax(parameters, LearningRate, BetaGradient, BetaSquaredGradient, Eps, WeightDecay));
    }
}
