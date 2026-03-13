using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace Bonsai.ML.Torch.NeuralNets.Optimizer;

/// <summary>
/// Represents an operator that creates a resilient backpropagation (Rprop) optimizer.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html"/> for more information.
/// </remarks>
[Description("Creates a resilient backpropagation (Rprop) optimizer.")]
[DisplayName("Rprop")]
public class ResilientBackpropagation
{
    /// <summary>
    /// The learning rate.
    /// </summary>
    [Description("The learning rate.")]
    public double LearningRate { get; set; } = 0.01D;

    /// <summary>
    /// The eta (-) parameter, which serves as a multiplicative factor to decrease the step size.
    /// </summary>
    [Description("The eta (-) parameter, which serves as a multiplicative factor to decrease the step size.")]
    public double EtaMinus { get; set; } = 0.5D;

    /// <summary>
    /// The eta (+) parameter, which serves as a multiplicative factor to increase the step size.
    /// </summary>
    [Description("The eta (+) parameter, which serves as a multiplicative factor to increase the step size.")]
    public double EtaPlus { get; set; } = 1.2D;

    /// <summary>
    /// The minimum allowed step size.
    /// </summary>
    [Description("The minimum allowed step size.")]
    public double MinStep { get; set; } = 1E-06D;

    /// <summary>
    /// The maximum allowed step size.
    /// </summary>
    [Description("The maximum allowed step size.")]
    public double MaxStep { get; set; } = 50D;

    /// <summary>
    /// If set to true, performs maximization instead of minimization of the params based on the objective.
    /// </summary>
    [Description("If set to true, performs maximization instead of minimization of the params based on the objective.")]
    public bool Maximize { get; set; } = false;

    /// <summary>
    /// Creates an Rprop optimizer from the input parameter collection.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Rprop> Process<T>(IObservable<T> source) where T : IEnumerable<Parameter>
    {
        return source.Select(parameters => Rprop(parameters, LearningRate, EtaMinus, EtaPlus, MinStep, MaxStep, Maximize));
    }
}
