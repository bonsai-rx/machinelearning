using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets.Optimizer;

/// <summary>
/// Represents an operator that creates an AdamW optimizer.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html"/> for more information.
/// </remarks>
[Description("Creates an AdamW optimizer.")]
public class AdamW
{
    /// <summary>
    /// The learning rate.
    /// </summary>
    [Description("The learning rate.")]
    public double LearningRate { get; set; } = 0.001D;

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
    /// If set to true, uses the AMSGrad variant of this algorithm.
    /// </summary>
    [Description("If set to true, uses the AMSGrad variant of this algorithm.")]
    public bool Amsgrad { get; set; } = false;

    /// <summary>
    /// If set to true, performs maximization instead of minimization of the params based on the objective.
    /// </summary>
    [Description("If set to true, performs maximization instead of minimization of the params based on the objective.")]
    public bool Maximize { get; set; } = false;

    /// <summary>
    /// Creates an AdamW optimizer from the input parameter collection.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<optim.Optimizer> Process<T>(IObservable<T> source) where T : IEnumerable<Parameter>
    {
        return source.Select(parameters => AdamW(parameters, LearningRate, BetaGradient, BetaSquaredGradient, Eps, WeightDecay, Amsgrad, Maximize));
    }
}
