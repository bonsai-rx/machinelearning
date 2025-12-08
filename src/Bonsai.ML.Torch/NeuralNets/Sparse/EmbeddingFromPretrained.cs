using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Sparse;

/// <summary>
/// Represents an operator that creates an embedding module from pretrained weights.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html"/> for more information.
/// </remarks>
[Description("Creates an embedding module from pretrained weights.")]
[TypeConverter(typeof(TensorOperatorConverter))]
public class EmbeddingFromPretrained : IScalarTypeProvider
{
    /// <summary>
    /// The pretrained weights for the embedding module.
    /// </summary>
    [XmlIgnore]
    [Description("The pretrained weights for the embedding module.")]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Embeddings { get; set; }

    /// <summary>
    /// The values of the embeddings tensor in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Embeddings))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string EmbeddingsXml
    {
        get => TensorConverter.ConvertToString(Embeddings, Type);
        set => Embeddings = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Determines whether to freeze the embeddings weights.
    /// </summary>
    [Description("Determines whether to freeze the embeddings weights.")]
    public bool Freeze { get; set; } = true;

    /// <summary>
    /// If specified, the entries do not contribute to the gradient and are not updated during training.
    /// </summary>
    [Description("If specified, the entries do not contribute to the gradient and are not updated during training.")]
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
    /// If set to true, the embeddings vectors are scaled by the inverse of their frequency in the input.
    /// </summary>
    [Description("If set to true, the embeddings vectors are scaled by the inverse of their frequency in the input.")]
    public bool ScaleGradByFreq { get; set; } = false;

    /// <summary>
    /// If set to true, the gradient will be sparse.
    /// </summary>
    [Description("If set to true, the gradient will be sparse.")]
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
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Creates an embedding module from pretrained weights.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Embedding_from_pretrained(Embeddings, Freeze, PaddingIdx, MaxNorm, NormType, ScaleGradByFreq, Sparse, Device, Type));
    }

    /// <summary>
    /// Creates an embedding module from pretrained weights.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Embedding_from_pretrained(Embeddings, Freeze, PaddingIdx, MaxNorm, NormType, ScaleGradByFreq, Sparse, Device, Type));
    }
}
