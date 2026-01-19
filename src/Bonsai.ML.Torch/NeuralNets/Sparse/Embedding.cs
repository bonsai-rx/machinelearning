using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Sparse;

/// <summary>
/// Represents an operator that creates an embedding module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html"/> for more information.
/// </remarks>
[Description("Creates an embedding module.")]
public class Embedding
{
    /// <summary>
    /// The size of the dictionary of embeddings.
    /// </summary>
    [Description("The size of the dictionary of embeddings.")]
    public long NumEmbeddings { get; set; }

    /// <summary>
    /// The size of each embedding vector.
    /// </summary>
    [Description("The size of each embedding vector.")]
    public long EmbeddingDims { get; set; }

    /// <summary>
    /// If specified, the padding indices do not contribute to the gradient and are not updated during training.
    /// </summary>
    [Description("If specified, the padding indices do not contribute to the gradient and are not updated during training.")]
    public long? PaddingIdx { get; set; } = null;

    /// <summary>
    /// If specified, each embedding vector is clipped to have this as its maximum norm.
    /// </summary>
    [Description("If specified, each embedding vector is clipped to have this as its maximum norm.")]
    public double? MaxNorm { get; set; } = null;

    /// <summary>
    /// The degree of the norm to compute for the max norm.
    /// </summary>
    [Description("The degree of the norm to compute for the max norm.")]
    public double NormType { get; set; } = 2D;

    /// <summary>
    /// If true, the embedding vectors are scaled by the inverse of their frequency in the input.
    /// </summary>
    [Description("If true, the embedding vectors are scaled by the inverse of their frequency in the input.")]
    public bool ScaleGradByFreq { get; set; } = false;

    /// <summary>
    /// If true, the gradient will be sparse.
    /// </summary>
    [Description("If true, the gradient will be sparse.")]
    public bool Sparse { get; set; } = false;

    /// <summary>
    /// The desired device of the returned tensor.
    /// </summary>
    [XmlIgnore]
    [Description("The desired device of the returned tensor.")]
    public Device Device { get; set; } = null;

    /// <summary>
    /// The desired data type of the returned tensor.
    /// </summary>
    [Description("The desired data type of the returned tensor.")]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Creates an embedding module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Embedding> Process()
    {
        return Observable.Return(Embedding(NumEmbeddings, EmbeddingDims, PaddingIdx, MaxNorm, NormType, ScaleGradByFreq, Sparse, Device, Type));
    }

    /// <summary>
    /// Creates an embedding module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Embedding> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Embedding(NumEmbeddings, EmbeddingDims, PaddingIdx, MaxNorm, NormType, ScaleGradByFreq, Sparse, Device, Type));
    }
}
