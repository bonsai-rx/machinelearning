using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace Bonsai.ML.Torch.NeuralNets.LearningRateScheduler;

/// <summary>
/// Represents an operator that creates a multi-step learning rate scheduler.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html"/> for more information.
/// </remarks>
[Description("Creates a multi-step learning rate scheduler.")]
public class MultiStep
{
    /// <summary>
    /// The list of epoch indices.
    /// </summary>
    [Description("The list of epoch indices.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public int[] Milestones { get; set; }

    /// <summary>
    /// The multiplicative factor of learning rate decay.
    /// </summary>
    [Description("The multiplicative factor of learning rate decay.")]
    public double Gamma { get; set; } = 0.1D;

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
    /// Creates a MultiStepLR scheduler.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LRScheduler> Process<T>(IObservable<T> source) where T : optim.Optimizer
    {
        return source.Select(optimizer => MultiStepLR(optimizer, Milestones, Gamma, LastEpoch, Verbose));
    }
}
