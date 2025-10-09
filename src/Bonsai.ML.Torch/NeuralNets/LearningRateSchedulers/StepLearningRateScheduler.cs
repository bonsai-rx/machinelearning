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
/// Creates a StepLearningRate scheduler.
/// </summary>
[Combinator]
[Description("Creates a StepLearningRate scheduler.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class StepLearningRateScheduler
{
    /// <summary>
    /// The optimizer parameter for the StepLR module.
    /// </summary>
    [Description("The optimizer parameter for the StepLR module")]
    public torch.optim.Optimizer Optimizer { get; set; }

    /// <summary>
    /// The step_size parameter for the StepLR module.
    /// </summary>
    [Description("The step_size parameter for the StepLR module")]
    public int StepSize { get; set; }

    /// <summary>
    /// The gamma parameter for the StepLR module.
    /// </summary>
    [Description("The gamma parameter for the StepLR module")]
    public double Gamma { get; set; } = 0.1D;

    /// <summary>
    /// The last_epoch parameter for the StepLR module.
    /// </summary>
    [Description("The last_epoch parameter for the StepLR module")]
    public int LastEpoch { get; set; } = -1;

    /// <summary>
    /// The verbose parameter for the StepLR module.
    /// </summary>
    [Description("The verbose parameter for the StepLR module")]
    public bool Verbose { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a StepLearningRateScheduler.
    /// </summary>
    public IObservable<LRScheduler> Process()
    {
        return Observable.Return(StepLR(Optimizer, StepSize, Gamma, LastEpoch, Verbose));
    }
}
