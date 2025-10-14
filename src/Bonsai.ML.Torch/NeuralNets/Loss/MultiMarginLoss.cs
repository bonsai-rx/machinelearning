using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Creates a MultiMarginLoss module.
/// </summary>
[Combinator]
[Description("Creates a MultiMarginLoss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MultiMarginLoss
{
    /// <summary>
    /// The p parameter for the MultiMarginLoss module.
    /// </summary>
    [Description("The p parameter for the MultiMarginLoss module")]
    public int P { get; set; } = 1;

    /// <summary>
    /// The margin parameter for the MultiMarginLoss module.
    /// </summary>
    [Description("The margin parameter for the MultiMarginLoss module")]
    public double Margin { get; set; } = 1D;

    /// <summary>
    /// The weight parameter for the MultiMarginLoss module.
    /// </summary>
    [Description("The weight parameter for the MultiMarginLoss module")]
    public torch.Tensor Weight { get; set; } = null;

    /// <summary>
    /// The reduction parameter for the MultiMarginLoss module.
    /// </summary>
    [Description("The reduction parameter for the MultiMarginLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a MultiMarginLoss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(MultiMarginLoss(P, Margin, Weight, Reduction));
    }
}
