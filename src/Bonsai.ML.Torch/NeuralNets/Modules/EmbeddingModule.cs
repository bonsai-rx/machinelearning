using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Modules;

/// <summary>
/// Creates a Embedding layer module.
/// </summary>
[Combinator]
[Description("Creates a Embedding layer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class EmbeddingModule
{
    /// <summary>
    /// The num_embeddings parameter for the Embedding module.
    /// </summary>
    [Description("The num_embeddings parameter for the Embedding module")]
    public long NumEmbeddings { get; set; }

    /// <summary>
    /// The embedding_dims parameter for the Embedding module.
    /// </summary>
    [Description("The embedding_dims parameter for the Embedding module")]
    public long EmbeddingDims { get; set; }

    /// <summary>
    /// The padding_idx parameter for the Embedding module.
    /// </summary>
    [Description("The padding_idx parameter for the Embedding module")]
    public long? PaddingIdx { get; set; } = null;

    /// <summary>
    /// The max_norm parameter for the Embedding module.
    /// </summary>
    [Description("The max_norm parameter for the Embedding module")]
    public double? MaxNorm { get; set; } = null;

    /// <summary>
    /// The norm_type parameter for the Embedding module.
    /// </summary>
    [Description("The norm_type parameter for the Embedding module")]
    public double NormType { get; set; } = 2;

    /// <summary>
    /// The scale_grad_by_freq parameter for the Embedding module.
    /// </summary>
    [Description("The scale_grad_by_freq parameter for the Embedding module")]
    public bool ScaleGradByFreq { get; set; } = false;

    /// <summary>
    /// The sparse parameter for the Embedding module.
    /// </summary>
    [Description("The sparse parameter for the Embedding module")]
    public bool Sparse { get; set; } = false;

    /// <summary>
    /// The desired device of returned tensor.
    /// </summary>
    [Description("The desired device of returned tensor")]
    [XmlIgnore]
    public Device Device { get; set; } = null;

    /// <summary>
    /// The desired data type of returned tensor.
    /// </summary>
    [Description("The desired data type of returned tensor")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a Embedding module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Embedding(NumEmbeddings, EmbeddingDims, PaddingIdx, MaxNorm, NormType, ScaleGradByFreq, Sparse, Device, Type));
    }
}
