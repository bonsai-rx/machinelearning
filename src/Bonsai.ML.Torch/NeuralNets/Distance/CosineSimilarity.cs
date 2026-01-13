using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Distance;

/// <summary>
/// Represents an operator that creates a cosine similarity module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html"/> for more information.
/// </remarks>
[Description("Creates a cosine similarity module.")]
public class CosineSimilarity
{
    /// <summary>
    /// The dimension where cosine similarity is computed.
    /// </summary>
    [Description("The dimension where cosine similarity is computed.")]
    public long Dim { get; set; } = 1;

    /// <summary>
    /// The value added to the denominator to avoid division by zero.
    /// </summary>
    [Description("The value added to the denominator to avoid division by zero.")]
    public double Eps { get; set; } = 1E-08D;

    /// <summary>
    /// Creates a CosineSimilarity module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(CosineSimilarity(Dim, Eps));
    }

    /// <summary>
    /// Creates a CosineSimilarity module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => CosineSimilarity(Dim, Eps));
    }
}
