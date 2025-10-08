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
/// Creates a ReduceLROnPlateau learning rate scheduler module.
/// </summary>
[Combinator]
[Description("Creates a ReduceLROnPlateau learning rate scheduler module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ReduceLROnPlateauLRSchedulerModule
{
    /// <summary>
    /// The optimizer parameter for the ReduceLROnPlateau module.
    /// </summary>
    [Description("The optimizer parameter for the ReduceLROnPlateau module")]
    public Optimizer Optimizer { get; set; }

    /// <summary>
    /// The mode parameter for the ReduceLROnPlateau module.
    /// </summary>
    [Description("The mode parameter for the ReduceLROnPlateau module")]
    public string Mode { get; set; } = "min";

    /// <summary>
    /// The factor parameter for the ReduceLROnPlateau module.
    /// </summary>
    [Description("The factor parameter for the ReduceLROnPlateau module")]
    public double Factor { get; set; } = 0.1;

    /// <summary>
    /// The patience parameter for the ReduceLROnPlateau module.
    /// </summary>
    [Description("The patience parameter for the ReduceLROnPlateau module")]
    public int Patience { get; set; } = 10;

    /// <summary>
    /// The threshold parameter for the ReduceLROnPlateau module.
    /// </summary>
    [Description("The threshold parameter for the ReduceLROnPlateau module")]
    public double Threshold { get; set; } = 0.0001;

    /// <summary>
    /// The threshold_mode parameter for the ReduceLROnPlateau module.
    /// </summary>
    [Description("The threshold_mode parameter for the ReduceLROnPlateau module")]
    public string ThresholdMode { get; set; } = "rel";

    /// <summary>
    /// The cooldown parameter for the ReduceLROnPlateau module.
    /// </summary>
    [Description("The cooldown parameter for the ReduceLROnPlateau module")]
    public int Cooldown { get; set; } = 0;

    /// <summary>
    /// The min_lr parameter for the ReduceLROnPlateau module.
    /// </summary>
    [Description("The min_lr parameter for the ReduceLROnPlateau module")]
    public IList<double> MinLr { get; set; } = null;

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-08;

    /// <summary>
    /// The verbose parameter for the ReduceLROnPlateau module.
    /// </summary>
    [Description("The verbose parameter for the ReduceLROnPlateau module")]
    public bool Verbose { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a ReduceLROnPlateauLRScheduler module.
    /// </summary>
    public IObservable<LRScheduler> Process()
    {
        return Observable.Return(ReduceLROnPlateau(Optimizer, Mode, Factor, Patience, Threshold, ThresholdMode, Cooldown, MinLr, Eps, Verbose));
    }
}
