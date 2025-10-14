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

namespace Bonsai.ML.Torch.NeuralNets.LinearRateScheduler;

/// <summary>
/// Creates a scheduler that decays the learning rate of each parameter group by gamma every step_size epochs.
/// </summary>
[Combinator]
[Description("Creates a scheduler that decays the learning rate of each parameter group by gamma every step_size epochs.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Step
{
    /// <summary>
    /// The optimizer parameter for the StepLR module.
    /// </summary>
    [Description("The optimizer parameter for the StepLR module")]
    public optim.Optimizer Optimizer { get; set; }

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
    /// Generates an observable sequence that creates a Step.
    /// </summary>
    public IObservable<LRScheduler> Process()
    {
        return Observable.Return(StepLR(Optimizer, StepSize, Gamma, LastEpoch, Verbose));
    }
}
