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
/// Creates a PolynomialLR learning rate scheduler module.
/// </summary>
[Combinator]
[Description("Creates a PolynomialLR learning rate scheduler module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PolynomialLRLRSchedulerModule
{
    /// <summary>
    /// The optimizer parameter for the PolynomialLR module.
    /// </summary>
    [Description("The optimizer parameter for the PolynomialLR module")]
    public Optimizer Optimizer { get; set; }

    /// <summary>
    /// The total_iters parameter for the PolynomialLR module.
    /// </summary>
    [Description("The total_iters parameter for the PolynomialLR module")]
    public int TotalIters { get; set; } = 5;

    /// <summary>
    /// The power parameter for the PolynomialLR module.
    /// </summary>
    [Description("The power parameter for the PolynomialLR module")]
    public int Power { get; set; } = 1;

    /// <summary>
    /// The last_epoch parameter for the PolynomialLR module.
    /// </summary>
    [Description("The last_epoch parameter for the PolynomialLR module")]
    public int LastEpoch { get; set; } = -1;

    /// <summary>
    /// The verbose parameter for the PolynomialLR module.
    /// </summary>
    [Description("The verbose parameter for the PolynomialLR module")]
    public bool Verbose { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a PolynomialLRLRScheduler module.
    /// </summary>
    public IObservable<LRScheduler> Process()
    {
        return Observable.Return(PolynomialLR(Optimizer, TotalIters, Power, LastEpoch, Verbose));
    }
}
