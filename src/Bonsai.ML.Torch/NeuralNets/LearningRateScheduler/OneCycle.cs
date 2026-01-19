using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace Bonsai.ML.Torch.NeuralNets.LearningRateScheduler;

/// <summary>
/// Represents an operator that creates a one cycle learning rate scheduler.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html"/> for more information.
/// </remarks>
[Description("Creates a one cycle learning rate scheduler.")]
public class OneCycle
{
    /// <summary>
    /// The upper learning rate boundaries in the cycle for each parameter group.
    /// </summary>
    [Description("The upper learning rate boundaries in the cycle for each parameter group.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public double[] MaxLearningRate { get; set; }

    /// <summary>
    /// The total number of steps in the cycle.
    /// </summary>
    [Description("The total number of steps in the cycle.")]
    public int TotalSteps { get; set; } = -1;

    /// <summary>
    /// The number of epochs to train for.
    /// </summary>
    [Description("The number of epochs to train for.")]
    public int Epochs { get; set; } = -1;

    /// <summary>
    /// The number of steps per epoch.
    /// </summary>
    [Description("The number of steps per epoch.")]
    public int StepsPerEpoch { get; set; } = -1;

    /// <summary>
    /// The percentage of the cycle spent increasing the learning rate.
    /// </summary>
    [Description("The percentage of the cycle spent increasing the learning rate.")]
    public double PercentageStart { get; set; } = 0.3D;

    /// <summary>
    /// The annealing strategy to use.
    /// </summary>
    [Description("The annealing strategy to use.")]
    public impl.OneCycleLR.AnnealStrategy AnnealStrategy { get; set; } = impl.OneCycleLR.AnnealStrategy.Cos;

    /// <summary>
    /// If true, momentum is cycled inversely to learning rate.
    /// </summary>
    [Description("If true, momentum is cycled inversely to learning rate.")]
    public bool CycleMomentum { get; set; } = true;

    /// <summary>
    /// The lower momentum boundaries in the cycle for each parameter group
    /// </summary>
    [Description("The lower momentum boundaries in the cycle for each parameter group")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public double[] BaseMomentum { get; set; } = null;

    /// <summary>
    /// The upper momentum boundaries in the cycle for each parameter group
    /// </summary>
    [Description("The upper momentum boundaries in the cycle for each parameter group")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public double[] MaxMomentum { get; set; } = null;

    /// <summary>
    /// Determines the initial learning rate.
    /// </summary>
    [Description("Determines the initial learning rate.")]
    public double DivFactor { get; set; } = 25D;

    /// <summary>
    /// Determines the final learning rate.
    /// </summary>
    [Description("Determines the final learning rate.")]
    public double FinalDivFactor { get; set; } = 10000D;

    /// <summary>
    /// If true, uses a third phase of the schedule to annihilate the learning rate.
    /// </summary>
    [Description("If true, uses a third phase of the schedule to annihilate the learning rate.")]
    public bool ThreePhase { get; set; } = false;

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
    /// Creates a OneCycleLR scheduler.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LRScheduler> Process<T>(IObservable<T> source) where T : optim.Optimizer
    {
        return source.Select(optimizer => OneCycleLR(optimizer, MaxLearningRate, TotalSteps, Epochs, StepsPerEpoch, PercentageStart, AnnealStrategy, CycleMomentum, BaseMomentum, MaxMomentum, DivFactor, FinalDivFactor, ThreePhase, LastEpoch, Verbose));
    }
}
