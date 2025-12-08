using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace Bonsai.ML.Torch.NeuralNets.LearningRateScheduler;

/// <summary>
/// Represents an operator that creates a constant learning rate scheduler.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ConstantLR.html"/> for more information.
/// </remarks>
[Description("Creates a constant learning rate scheduler.")]
public class Constant
{
    /// <summary>
    /// The number that the learning rate will be multiplied by until the milestone.
    /// </summary>
    [Description("The number that the learning rate will be multiplied by until the milestone.")]
    public double Factor { get; set; } = 0.3333333333333333D;

    /// <summary>
    /// The number of steps that the scheduler multiplies the learning rate by the factor.
    /// </summary>
    [Description("The number of steps that the scheduler multiplies the learning rate by the factor.")]
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
    /// Creates a ConstantLR scheduler for the input optimizer.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LRScheduler> Process<T>(IObservable<T> source) where T : optim.Optimizer
    {
        return source.Select(optimizer => ConstantLR(optimizer, Factor, TotalIters, LastEpoch, Verbose));
    }
}
