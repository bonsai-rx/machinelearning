using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace Bonsai.ML.Torch.NeuralNets.LearningRateScheduler;

/// <summary>
/// Represents an operator that creates a polynomial learning rate scheduler.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.PolynomialLR.html"/> for more information.
/// </remarks>
[Description("Creates a polynomial learning rate scheduler.")]
public class Polynomial
{
    /// <summary>
    /// The total number of steps that the learning rate will be adjusted over.
    /// </summary>
    [Description("The total number of steps that the learning rate will be adjusted over.")]
    public int TotalIters { get; set; } = 5;

    /// <summary>
    /// The power of the polynomial.
    /// </summary>
    [Description("The power of the polynomial.")]
    public int Power { get; set; } = 1;

    /// <summary>
    /// The index of the last epoch.
    /// </summary>
    [Description("The index of the last epoch.")]
    public int LastEpoch { get; set; } = -1;

    /// <summary>
    /// Determines whether to write a message to stdout for each update.
    /// </summary>
    [Description("Determines whether to write a message to stdout for each update.")]
    public bool Verbose { get; set; } = false;

    /// <summary>
    /// Creates a PolynomialLR scheduler.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LRScheduler> Process<T>(IObservable<T> source) where T : optim.Optimizer
    {
        return source.Select(optimizer => PolynomialLR(optimizer, TotalIters, Power, LastEpoch, Verbose));
    }
}
