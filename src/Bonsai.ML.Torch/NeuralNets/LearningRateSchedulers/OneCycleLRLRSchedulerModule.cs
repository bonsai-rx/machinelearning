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
/// Creates a OneCycleLR learning rate scheduler module.
/// </summary>
[Combinator]
[Description("Creates a OneCycleLR learning rate scheduler module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class OneCycleLRLRSchedulerModule
{
    /// <summary>
    /// The optimizer parameter for the OneCycleLR module.
    /// </summary>
    [Description("The optimizer parameter for the OneCycleLR module")]
    public Optimizer Optimizer { get; set; }

    /// <summary>
    /// The max_lr parameter for the OneCycleLR module.
    /// </summary>
    [Description("The max_lr parameter for the OneCycleLR module")]
    public double MaxLr { get; set; }

    /// <summary>
    /// The total_steps parameter for the OneCycleLR module.
    /// </summary>
    [Description("The total_steps parameter for the OneCycleLR module")]
    public int TotalSteps { get; set; } = -1;

    /// <summary>
    /// The epochs parameter for the OneCycleLR module.
    /// </summary>
    [Description("The epochs parameter for the OneCycleLR module")]
    public int Epochs { get; set; } = -1;

    /// <summary>
    /// The steps_per_epoch parameter for the OneCycleLR module.
    /// </summary>
    [Description("The steps_per_epoch parameter for the OneCycleLR module")]
    public int StepsPerEpoch { get; set; } = -1;

    /// <summary>
    /// The pct_start parameter for the OneCycleLR module.
    /// </summary>
    [Description("The pct_start parameter for the OneCycleLR module")]
    public double PctStart { get; set; } = 0.3;

    /// <summary>
    /// The anneal_strategy parameter for the OneCycleLR module.
    /// </summary>
    [Description("The anneal_strategy parameter for the OneCycleLR module")]
    public impl.OneCycleLR.AnnealStrategy AnnealStrategy { get; set; } = impl.OneCycleLR.AnnealStrategy.Cos;

    /// <summary>
    /// The cycle_momentum parameter for the OneCycleLR module.
    /// </summary>
    [Description("The cycle_momentum parameter for the OneCycleLR module")]
    public bool CycleMomentum { get; set; } = true;

    /// <summary>
    /// The base_momentum parameter for the OneCycleLR module.
    /// </summary>
    [Description("The base_momentum parameter for the OneCycleLR module")]
    public double BaseMomentum { get; set; } = 0.85;

    /// <summary>
    /// The max_momentum parameter for the OneCycleLR module.
    /// </summary>
    [Description("The max_momentum parameter for the OneCycleLR module")]
    public double MaxMomentum { get; set; } = 0.95;

    /// <summary>
    /// The div_factor parameter for the OneCycleLR module.
    /// </summary>
    [Description("The div_factor parameter for the OneCycleLR module")]
    public double DivFactor { get; set; } = 25;

    /// <summary>
    /// The final_div_factor parameter for the OneCycleLR module.
    /// </summary>
    [Description("The final_div_factor parameter for the OneCycleLR module")]
    public double FinalDivFactor { get; set; } = 10000;

    /// <summary>
    /// The three_phase parameter for the OneCycleLR module.
    /// </summary>
    [Description("The three_phase parameter for the OneCycleLR module")]
    public bool ThreePhase { get; set; } = false;

    /// <summary>
    /// The last_epoch parameter for the OneCycleLR module.
    /// </summary>
    [Description("The last_epoch parameter for the OneCycleLR module")]
    public int LastEpoch { get; set; } = -1;

    /// <summary>
    /// The verbose parameter for the OneCycleLR module.
    /// </summary>
    [Description("The verbose parameter for the OneCycleLR module")]
    public bool Verbose { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a OneCycleLRLRScheduler module.
    /// </summary>
    public IObservable<LRScheduler> Process()
    {
        return Observable.Return(OneCycleLR(Optimizer, MaxLr, TotalSteps, Epochs, StepsPerEpoch, PctStart, AnnealStrategy, CycleMomentum, BaseMomentum, MaxMomentum, DivFactor, FinalDivFactor, ThreePhase, LastEpoch, Verbose));
    }
}
