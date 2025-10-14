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
/// Creates a MultiLabelSoftMarginLoss module.
/// </summary>
[Combinator]
[Description("Creates a MultiLabelSoftMarginLoss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MultiLabelSoftMargin
{
    /// <summary>
    /// The weight parameter for the MultiLabelSoftMarginLoss module.
    /// </summary>
    [Description("The weight parameter for the MultiLabelSoftMarginLoss module")]
    public torch.Tensor Weight { get; set; } = null;

    /// <summary>
    /// The reduction parameter for the MultiLabelSoftMarginLoss module.
    /// </summary>
    [Description("The reduction parameter for the MultiLabelSoftMarginLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a MultiLabelSoftMarginLoss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(MultiLabelSoftMarginLoss(Weight, Reduction));
    }
}
