using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace Bonsai.ML.Torch.NeuralNets.LearningRateScheduler;

/// <summary>
/// Represents an operator that creates a linear learning rate scheduler.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html"/> for more information.
/// </remarks>
[Description("Creates a linear learning rate scheduler.")]
public class Linear
{
    /// <summary>
    /// The number used to multiply the learning rate in the first epoch.
    /// </summary>
    [Description("The number used to multiply the learning rate in the first epoch.")]
    public double StartFactor { get; set; } = 0.3333333333333333D;

    /// <summary>
    /// The number used to multiply the learning rate in the last epoch.
    /// </summary>
    [Description("The number used to multiply the learning rate in the last epoch.")]
    public double EndFactor { get; set; } = 5D;

    /// <summary>
    /// The number of iterations over which the learning rate is adjusted.
    /// </summary>
    [Description("The number of iterations over which the learning rate is adjusted.")]
    public int TotalIters { get; set; } = 5;

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
    /// Creates a LinearLR scheduler.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LRScheduler> Process<T>(IObservable<T> source) where T : optim.Optimizer
    {
        return source.Select(optimizer => LinearLR(optimizer, StartFactor, EndFactor, TotalIters, LastEpoch, Verbose));
    }
}
