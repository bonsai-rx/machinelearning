using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;

namespace Bonsai.ML.Torch.NeuralNets.LearningRateSchedulers;

/// <summary>
/// Creates a LinearLR learning rate scheduler module.
/// </summary>
[Combinator]
[Description("Creates a LinearLR learning rate scheduler module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LinearLRLRSchedulerModule
{
    /// <summary>
    /// The optimizer parameter for the LinearLR module.
    /// </summary>
    [Description("The optimizer parameter for the LinearLR module")]
    public Optimizer Optimizer { get; set; }

    /// <summary>
    /// The start_factor parameter for the LinearLR module.
    /// </summary>
    [Description("The start_factor parameter for the LinearLR module")]
    public double StartFactor { get; set; } = 0.3333333333333333;

    /// <summary>
    /// The end_factor parameter for the LinearLR module.
    /// </summary>
    [Description("The end_factor parameter for the LinearLR module")]
    public double EndFactor { get; set; } = 5;

    /// <summary>
    /// The total_iters parameter for the LinearLR module.
    /// </summary>
    [Description("The total_iters parameter for the LinearLR module")]
    public int TotalIters { get; set; } = 5;

    /// <summary>
    /// The last_epoch parameter for the LinearLR module.
    /// </summary>
    [Description("The last_epoch parameter for the LinearLR module")]
    public int LastEpoch { get; set; } = -1;

    /// <summary>
    /// The verbose parameter for the LinearLR module.
    /// </summary>
    [Description("The verbose parameter for the LinearLR module")]
    public bool Verbose { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a LinearLRLRScheduler module.
    /// </summary>
    public IObservable<LRScheduler> Process()
    {
        return Observable.Return(LinearLR(Optimizer, StartFactor, EndFactor, TotalIters, LastEpoch, Verbose));
    }
}
