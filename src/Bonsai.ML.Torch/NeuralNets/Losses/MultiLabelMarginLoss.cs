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
/// Creates a MultiLabelMarginLoss module.
/// </summary>
[Combinator]
[Description("Creates a MultiLabelMarginLoss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MultiLabelMarginLoss
{
    /// <summary>
    /// The reduction parameter for the MultiLabelMarginLoss module.
    /// </summary>
    [Description("The reduction parameter for the MultiLabelMarginLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a MultiLabelMarginLoss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(MultiLabelMarginLoss(Reduction));
    }
}
