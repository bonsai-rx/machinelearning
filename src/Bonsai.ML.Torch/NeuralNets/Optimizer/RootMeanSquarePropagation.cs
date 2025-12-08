using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets.Optimizer;

/// <summary>
/// Represents an operator that creates a root mean square propagation (RMSProp) optimizer.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html"/> for more information.
/// </remarks>
[Description("Creates a root mean square propagation (RMSProp) optimizer.")]
public class RootMeanSquarePropagation
{
    /// <summary>
    /// The learning rate.
    /// </summary>
    [Description("The learning rate.")]
    public double LearningRate { get; set; } = 0.01D;

    /// <summary>
    /// The alpha parameter, which acts as a smoothing constant.
    /// </summary>
    [Description("The alpha parameter, which acts as a smoothing constant.")]
    public double Alpha { get; set; } = 0.99D;

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
    /// The momentum factor, which accelerates learning in the relevant direction.
    /// </summary>
    [Description("The momentum factor, which accelerates learning in the relevant direction.")]
    public double Momentum { get; set; } = 0D;

    /// <summary>
    /// If set to true, the gradient is normalized by an estimation of its variance.
    /// </summary>
    [Description("If set to true, the gradient is normalized by an estimation of its variance.")]
    public bool Centered { get; set; } = false;

    /// <summary>
    /// If set to true, performs maximization instead of minimization of the params based on the objective.
    /// </summary>
    [Description("If set to true, performs maximization instead of minimization of the params based on the objective.")]
    public bool Maximize { get; set; } = false;

    /// <summary>
    /// Creates an RMSProp optimizer from the input parameter collection.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<optim.Optimizer> Process<T>(IObservable<T> source) where T : IEnumerable<Parameter>
    {
        return source.Select(parameters => RMSProp(parameters, LearningRate, Alpha, Eps, WeightDecay, Momentum, Centered, Maximize));
    }
}
