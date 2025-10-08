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
/// Creates a HingeEmbeddingLoss module module.
/// </summary>
[Combinator]
[Description("Creates a HingeEmbeddingLoss module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class HingeEmbeddingLossModule
{
    /// <summary>
    /// The margin parameter for the HingeEmbeddingLoss module.
    /// </summary>
    [Description("The margin parameter for the HingeEmbeddingLoss module")]
    public double Margin { get; set; } = 1;

    /// <summary>
    /// The reduction parameter for the HingeEmbeddingLoss module.
    /// </summary>
    [Description("The reduction parameter for the HingeEmbeddingLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a HingeEmbeddingLoss module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(HingeEmbeddingLoss(Margin, Reduction));
    }
}
