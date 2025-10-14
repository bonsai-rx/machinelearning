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

namespace Bonsai.ML.Torch.NeuralNets.LearningRateScheduler;

/// <summary>
/// Sets the learning rate using a cosine annealing schedule.
/// </summary>
[Combinator]
[Description("Sets the learning rate using a cosine annealing schedule.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CosineAnnealing
{
    /// <summary>
    /// The wrapped optimizer.
    /// </summary>
    [Description("The wrapped optimizer.")]
    public optim.Optimizer Optimizer { get; set; }

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
    /// The index
    /// </summary>
    [Description("The last_epoch parameter for the CosineAnnealing module")]
    public int LastEpoch { get; set; } = -1;

    /// <summary>
    /// The verbose parameter for the CosineAnnealing module.
    /// </summary>
    [Description("The verbose parameter for the CosineAnnealing module")]
    public bool Verbose { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a CosineAnnealingLearningRateScheduler.
    /// </summary>
    public IObservable<LRScheduler> Process()
    {
        return Observable.Return(CosineAnnealingLR(Optimizer, TMax, EtaMin, LastEpoch, Verbose));
    }
}
