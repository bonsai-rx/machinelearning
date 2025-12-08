using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace Bonsai.ML.Torch.NeuralNets.LearningRateScheduler;

/// <summary>
/// Represents an operator that creates a step learning rate scheduler.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html"/> for more information.
/// </remarks>
[Description("Creates a step learning rate scheduler.")]
public class Step
{
    /// <summary>
    /// The period of learning rate decay.
    /// </summary>
    [Description("The period of learning rate decay.")]
    public int StepSize { get; set; }

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
    /// Creates a StepLR scheduler.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LRScheduler> Process<T>(IObservable<T> source) where T : optim.Optimizer
    {
        return source.Select(optimizer => StepLR(optimizer, StepSize, Gamma, LastEpoch, Verbose));
    }
}
