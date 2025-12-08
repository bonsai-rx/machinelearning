using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace Bonsai.ML.Torch.NeuralNets.LearningRateScheduler;

/// <summary>
/// Represents an operator that creates a cosine annealing learning rate scheduler.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html"/> for more information.
/// </remarks>
[Description("Creates a cosine annealing learning rate scheduler.")]
public class CosineAnnealing
{
    /// <summary>
    /// The maximum number of iterations.
    /// </summary>
    [Description("The maximum number of iterations.")]
    public double TMax { get; set; }

    /// <summary>
    /// The minimum learning rate.
    /// </summary>
    [Description("The minimum learning rate.")]
    public double EtaMin { get; set; } = 0D;

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
    /// Creates a CosineAnnealingLR scheduler.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LRScheduler> Process<T>(IObservable<T> source) where T : optim.Optimizer
    {
        return source.Select(optimizer => CosineAnnealingLR(optimizer, TMax, EtaMin, LastEpoch, Verbose));
    }
}
