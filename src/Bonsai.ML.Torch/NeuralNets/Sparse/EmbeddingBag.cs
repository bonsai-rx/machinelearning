using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Sparse;

/// <summary>
/// Represents an operator that creates an embedding bag module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html"/> for more information.
/// </remarks>
[Description("Creates an embedding bag module.")]
public class EmbeddingBag
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
    /// If set to true, the embeddings vectors are scaled by the inverse of their frequency in the input.
    /// </summary>
    [Description("If set to true, the embeddings vectors are scaled by the inverse of their frequency in the input.")]
    public bool ScaleGradByFreq { get; set; } = false;

    /// <summary>
    /// The type of reduction to apply to the embeddings.
    /// </summary>
    [Description("The type of reduction to apply to the embeddings.")]
    public EmbeddingBagMode Mode { get; set; } = EmbeddingBagMode.Mean;

    /// <summary>
    /// If set to true, the gradient will be sparse.
    /// </summary>
    [Description("If set to true, the gradient will be sparse.")]
    public bool Sparse { get; set; } = false;

    /// <summary>
    /// If set to true, the offsets tensor includes an additional element at the end equal to the size of indices.
    /// </summary>
    [Description("If set to true, the offsets tensor includes an additional element at the end equal to the size of indices.")]
    public bool IncludeLastOffset { get; set; } = false;

    /// <summary>
    /// If specified, the entries do not contribute to the gradient and are not updated during training.
    /// </summary>
    [Description("If specified, the entries do not contribute to the gradient and are not updated during training.")]
    public long PaddingIndex { get; set; } = -1;

    /// <summary>
    /// The desired device of the returned tensor.
    /// </summary>
    [XmlIgnore]
    [Description("The desired device of the returned tensor")]
    public Device Device { get; set; } = null;

    /// <summary>
    /// The desired data type of the returned tensor.
    /// </summary>
    [Description("The desired data type of the returned tensor")]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Creates an embedding bag module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.EmbeddingBag> Process()
    {
        return Observable.Return(EmbeddingBag(NumEmbeddings, EmbeddingDims, MaxNorm, NormType, ScaleGradByFreq, Mode, Sparse, IncludeLastOffset, PaddingIndex, Device, Type));
    }

    /// <summary>
    /// Creates an embedding bag module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.EmbeddingBag> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => EmbeddingBag(NumEmbeddings, EmbeddingDims, MaxNorm, NormType, ScaleGradByFreq, Mode, Sparse, IncludeLastOffset, PaddingIndex, Device, Type));
    }
}
