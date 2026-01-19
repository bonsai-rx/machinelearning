using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace Bonsai.ML.Torch.NeuralNets.LearningRateScheduler;

/// <summary>
/// Represents an operator that creates a scheduler that reduces the learning rate on plateau.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html"/> for more information.
/// </remarks>
[Description("Creates a scheduler that reduces the learning rate on plateau.")]
public class ReduceOnPlateau
{
    /// <summary>
    /// The mode in which to determine when to reduce the learning rate.
    /// </summary>
    [Description("The mode in which to determine when to reduce the learning rate.")]
    public PlateauMode Mode { get; set; } = PlateauMode.Min;

    /// <summary>
    /// The factor by which the learning rate will be reduced.
    /// </summary>
    [Description("The factor by which the learning rate will be reduced.")]
    public double Factor { get; set; } = 0.1D;

    /// <summary>
    /// The number of allowed epochs with no improvement after which the learning rate will be reduced.
    /// </summary>
    [Description("The number of allowed epochs with no improvement after which the learning rate will be reduced.")]
    public int Patience { get; set; } = 10;

    /// <summary>
    /// The threshold for measuring the new optimum, to only focus on significant changes.
    /// </summary>
    [Description("The threshold for measuring the new optimum, to only focus on significant changes.")]
    public double Threshold { get; set; } = 0.0001D;

    /// <summary>
    /// The mode for the threshold parameter.
    /// </summary>
    [Description("The mode for the threshold parameter.")]
    public PlateauThresholdMode ThresholdMode { get; set; } = PlateauThresholdMode.Relative;

    /// <summary>
    /// The number of epochs to wait before resuming normal operation after the learning rate has been reduced.
    /// </summary>
    [Description("The number of epochs to wait before resuming normal operation after the learning rate has been reduced.")]
    public int Cooldown { get; set; } = 0;

    /// <summary>
    /// The lower bound on the learning rate of all param groups or each group respectively.
    /// </summary>
    [Description("The lower bound on the learning rate of all param groups or each group respectively.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public double[] MinLearningRate { get; set; } = null;

    /// <summary>
    /// The minimal decay applied to the learning rate.
    /// </summary>
    [Description("The minimal decay applied to the learning rate")]
    public double Eps { get; set; } = 1E-08D;

    /// <summary>
    /// Determines whether to write a message to stdout for each update.
    /// </summary>
    [Description("Determines whether to write a message to stdout for each update.")]
    public bool Verbose { get; set; } = false;

#pragma warning disable // Ignore missing xml doc strings
    public enum PlateauMode
    {
        Min,
        Max
    }

    public enum PlateauThresholdMode
    {
        Relative,
        Absolute
    }

    private string ModeString => Mode == PlateauMode.Min ? "min" : "max";

    private string ThresholdModeString => ThresholdMode == PlateauThresholdMode.Relative ? "rel" : "abs";

#pragma warning restore

    /// <summary>
    /// Creates a ReduceLROnPlateau scheduler.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LRScheduler> Process<T>(IObservable<T> source) where T : optim.Optimizer
    {
        return source.Select(optimizer => ReduceLROnPlateau(optimizer, ModeString, Factor, Patience, Threshold, ThresholdModeString, Cooldown, MinLearningRate, Eps, Verbose));
    }
}
