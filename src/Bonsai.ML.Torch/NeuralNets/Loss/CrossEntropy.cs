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
/// Computes the cross entropy loss between input logits and target.
/// </summary>
[Combinator]
[Description("Computes the cross entropy loss between input logits and target.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CrossEntropy
{
    /// <summary>
    /// The weight parameter for the CrossEntropyLoss module.
    /// </summary>
    [Description("The weight parameter for the CrossEntropyLoss module")]
    public Tensor Weight { get; set; } = null;

    /// <summary>
    /// The ignore_index parameter for the CrossEntropyLoss module.
    /// </summary>
    [Description("The ignore_index parameter for the CrossEntropyLoss module")]
    public long? IgnoreIndex { get; set; } = null;

    /// <summary>
    /// The reduction parameter for the CrossEntropyLoss module.
    /// </summary>
    [Description("The reduction parameter for the CrossEntropyLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a CrossEntropyLoss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(CrossEntropyLoss(Weight, IgnoreIndex, Reduction));
    }
}
