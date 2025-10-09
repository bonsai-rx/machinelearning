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
/// Creates a CosineAnnealingLearningRate scheduler.
/// </summary>
[Combinator]
[Description("Creates a CosineAnnealingLearningRate scheduler.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CosineAnnealingLearningRateScheduler
{
    /// <summary>
    /// The optimizer parameter for the CosineAnnealingLR module.
    /// </summary>
    [Description("The optimizer parameter for the CosineAnnealingLR module")]
    public torch.optim.Optimizer Optimizer { get; set; }

    /// <summary>
    /// The t_max parameter for the CosineAnnealingLR module.
    /// </summary>
    [Description("The t_max parameter for the CosineAnnealingLR module")]
    public double TMax { get; set; }

    /// <summary>
    /// The eta_min parameter for the CosineAnnealingLR module.
    /// </summary>
    [Description("The eta_min parameter for the CosineAnnealingLR module")]
    public double EtaMin { get; set; } = 0D;

    /// <summary>
    /// The last_epoch parameter for the CosineAnnealingLR module.
    /// </summary>
    [Description("The last_epoch parameter for the CosineAnnealingLR module")]
    public int LastEpoch { get; set; } = -1;

    /// <summary>
    /// The verbose parameter for the CosineAnnealingLR module.
    /// </summary>
    [Description("The verbose parameter for the CosineAnnealingLR module")]
    public bool Verbose { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a CosineAnnealingLearningRateScheduler.
    /// </summary>
    public IObservable<LRScheduler> Process()
    {
        return Observable.Return(CosineAnnealingLR(Optimizer, TMax, EtaMin, LastEpoch, Verbose));
    }
}
