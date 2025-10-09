using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Losses;

/// <summary>
/// Creates a BCEWithLogitsLoss module.
/// </summary>
[Combinator]
[Description("Creates a BCEWithLogitsLoss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class BCEWithLogitsLoss
{
    /// <summary>
    /// The weight parameter for the BCEWithLogitsLoss module.
    /// </summary>
    [Description("The weight parameter for the BCEWithLogitsLoss module")]
    public torch.Tensor Weight { get; set; } = null;

    /// <summary>
    /// The reduction parameter for the BCEWithLogitsLoss module.
    /// </summary>
    [Description("The reduction parameter for the BCEWithLogitsLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// The pos_weights parameter for the BCEWithLogitsLoss module.
    /// </summary>
    [Description("The pos_weights parameter for the BCEWithLogitsLoss module")]
    public torch.Tensor PosWeights { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a BCEWithLogitsLoss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(BCEWithLogitsLoss(Weight, Reduction, PosWeights));
    }
}
