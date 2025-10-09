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
/// Creates a BCELoss module.
/// </summary>
[Combinator]
[Description("Creates a BCELoss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class BCELoss
{
    /// <summary>
    /// The weight parameter for the BCELoss module.
    /// </summary>
    [Description("The weight parameter for the BCELoss module")]
    public torch.Tensor Weight { get; set; } = null;

    /// <summary>
    /// The reduction parameter for the BCELoss module.
    /// </summary>
    [Description("The reduction parameter for the BCELoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a BCELoss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(BCELoss(Weight, Reduction));
    }
}
