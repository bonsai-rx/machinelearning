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
/// Creates a HingeEmbeddingLoss module.
/// </summary>
[Combinator]
[Description("Creates a HingeEmbeddingLoss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class HingeEmbeddingLoss
{
    /// <summary>
    /// The margin parameter for the HingeEmbeddingLoss module.
    /// </summary>
    [Description("The margin parameter for the HingeEmbeddingLoss module")]
    public double Margin { get; set; } = 1D;

    /// <summary>
    /// The reduction parameter for the HingeEmbeddingLoss module.
    /// </summary>
    [Description("The reduction parameter for the HingeEmbeddingLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a HingeEmbeddingLoss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(HingeEmbeddingLoss(Margin, Reduction));
    }
}
