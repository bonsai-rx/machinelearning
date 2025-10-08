using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Losses;

/// <summary>
/// Creates a CosineEmbeddingLoss module module.
/// </summary>
[Combinator]
[Description("Creates a CosineEmbeddingLoss module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CosineEmbeddingLossModule
{
    /// <summary>
    /// The margin parameter for the CosineEmbeddingLoss module.
    /// </summary>
    [Description("The margin parameter for the CosineEmbeddingLoss module")]
    public double Margin { get; set; } = 0;

    /// <summary>
    /// The reduction parameter for the CosineEmbeddingLoss module.
    /// </summary>
    [Description("The reduction parameter for the CosineEmbeddingLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a CosineEmbeddingLoss module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(CosineEmbeddingLoss(Margin, Reduction));
    }
}
