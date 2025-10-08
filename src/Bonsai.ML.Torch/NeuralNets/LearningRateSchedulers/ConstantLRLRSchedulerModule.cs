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
/// Creates a ConstantLR learning rate scheduler module.
/// </summary>
[Combinator]
[Description("Creates a ConstantLR learning rate scheduler module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ConstantLRLRSchedulerModule
{
    /// <summary>
    /// The optimizer parameter for the ConstantLR module.
    /// </summary>
    [Description("The optimizer parameter for the ConstantLR module")]
    public Optimizer Optimizer { get; set; }

    /// <summary>
    /// The factor parameter for the ConstantLR module.
    /// </summary>
    [Description("The factor parameter for the ConstantLR module")]
    public double Factor { get; set; } = 0.3333333333333333;

    /// <summary>
    /// The total_iters parameter for the ConstantLR module.
    /// </summary>
    [Description("The total_iters parameter for the ConstantLR module")]
    public int TotalIters { get; set; } = 5;

    /// <summary>
    /// The last_epoch parameter for the ConstantLR module.
    /// </summary>
    [Description("The last_epoch parameter for the ConstantLR module")]
    public int LastEpoch { get; set; } = -1;

    /// <summary>
    /// The verbose parameter for the ConstantLR module.
    /// </summary>
    [Description("The verbose parameter for the ConstantLR module")]
    public bool Verbose { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a ConstantLRLRScheduler module.
    /// </summary>
    public IObservable<LRScheduler> Process()
    {
        return Observable.Return(ConstantLR(Optimizer, Factor, TotalIters, LastEpoch, Verbose));
    }
}
