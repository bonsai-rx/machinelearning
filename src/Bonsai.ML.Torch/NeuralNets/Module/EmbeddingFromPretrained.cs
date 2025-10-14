using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Module;

/// <summary>
/// Creates a EmbeddingFromPretrained module.
/// </summary>
[Combinator]
[Description("Creates a EmbeddingFromPretrained module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class EmbeddingFromPretrained
{
    /// <summary>
    /// The embeddings parameter for the Embedding_from_pretrained module.
    /// </summary>
    [Description("The embeddings parameter for the Embedding_from_pretrained module")]
    public torch.Tensor Embeddings { get; set; }

    /// <summary>
    /// The freeze parameter for the Embedding_from_pretrained module.
    /// </summary>
    [Description("The freeze parameter for the Embedding_from_pretrained module")]
    public bool Freeze { get; set; } = true;

    /// <summary>
    /// The padding_idx parameter for the Embedding_from_pretrained module.
    /// </summary>
    [Description("The padding_idx parameter for the Embedding_from_pretrained module")]
    public long? PaddingIdx { get; set; } = null;

    /// <summary>
    /// The max_norm parameter for the Embedding_from_pretrained module.
    /// </summary>
    [Description("The max_norm parameter for the Embedding_from_pretrained module")]
    public double? MaxNorm { get; set; } = null;

    /// <summary>
    /// The norm_type parameter for the Embedding_from_pretrained module.
    /// </summary>
    [Description("The norm_type parameter for the Embedding_from_pretrained module")]
    public double NormType { get; set; } = 2D;

    /// <summary>
    /// The scale_grad_by_freq parameter for the Embedding_from_pretrained module.
    /// </summary>
    [Description("The scale_grad_by_freq parameter for the Embedding_from_pretrained module")]
    public bool ScaleGradByFreq { get; set; } = false;

    /// <summary>
    /// The sparse parameter for the Embedding_from_pretrained module.
    /// </summary>
    [Description("The sparse parameter for the Embedding_from_pretrained module")]
    public bool Sparse { get; set; } = false;

    /// <summary>
    /// The desired device of returned tensor.
    /// </summary>
    [Description("The desired device of returned tensor")]
    public torch.Device Device { get; set; } = null;

    /// <summary>
    /// The desired data type of returned tensor.
    /// </summary>
    [Description("The desired data type of returned tensor")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a EmbeddingFromPretrainedModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Embedding_from_pretrained(Embeddings, Freeze, PaddingIdx, MaxNorm, NormType, ScaleGradByFreq, Sparse, Device, Type));
    }
}
