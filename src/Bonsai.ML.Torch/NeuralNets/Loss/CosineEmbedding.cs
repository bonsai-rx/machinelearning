using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Represents an operator that creates a cosine embedding loss module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html"/> for more information.
/// </remarks>
[Description("Creates a cosine embedding loss module.")]
public class CosineEmbedding
{
    /// <summary>
    /// The margin, which is a value in the range [-1, 1].
    /// </summary>
    [Description("The margin, which is a value in the range [-1, 1].")]
    public double Margin { get; set; } = 0D;

    /// <summary>
    /// The reduction type to apply to the output.
    /// </summary>
    [Description("The reduction type to apply to the output.")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Creates a cosine embedding loss module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(CosineEmbeddingLoss(Margin, Reduction));
    }

    /// <summary>
    /// Creates a cosine embedding loss module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => CosineEmbeddingLoss(Margin, Reduction));
    }
}
