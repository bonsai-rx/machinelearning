using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Represents an operator that creates a hinge embedding loss module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html"/> for more information.
/// </remarks>
[Description("Creates a hinge embedding loss module.")]
public class HingeEmbedding
{
    /// <summary>
    /// The margin, a non-negative float value.
    /// </summary>
    [Description("The margin, a non-negative float value.")]
    public double Margin { get; set; } = 1D;

    /// <summary>
    /// The reduction type to apply to the output.
    /// </summary>
    [Description("The reduction type to apply to the output.")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Creates a hinge embedding loss module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.HingeEmbeddingLoss> Process()
    {
        return Observable.Return(HingeEmbeddingLoss(Margin, Reduction));
    }

    /// <summary>
    /// Creates a hinge embedding loss module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.HingeEmbeddingLoss> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => HingeEmbeddingLoss(Margin, Reduction));
    }
}
