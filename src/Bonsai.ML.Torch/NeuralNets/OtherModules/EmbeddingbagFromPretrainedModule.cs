using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.OtherModules;

/// <summary>
/// Creates a EmbeddingbagFromPretrained module.
/// </summary>
[Combinator]
[Description("Creates a EmbeddingbagFromPretrained module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class EmbeddingbagFromPretrainedModule
{
    /// <summary>
    /// The embeddings parameter for the EmbeddingBag_from_pretrained module.
    /// </summary>
    [Description("The embeddings parameter for the EmbeddingBag_from_pretrained module")]
    public torch.Tensor Embeddings { get; set; }

    /// <summary>
    /// The freeze parameter for the EmbeddingBag_from_pretrained module.
    /// </summary>
    [Description("The freeze parameter for the EmbeddingBag_from_pretrained module")]
    public bool Freeze { get; set; } = true;

    /// <summary>
    /// The max_norm parameter for the EmbeddingBag_from_pretrained module.
    /// </summary>
    [Description("The max_norm parameter for the EmbeddingBag_from_pretrained module")]
    public double? MaxNorm { get; set; } = null;

    /// <summary>
    /// The norm_type parameter for the EmbeddingBag_from_pretrained module.
    /// </summary>
    [Description("The norm_type parameter for the EmbeddingBag_from_pretrained module")]
    public double NormType { get; set; } = 2D;

    /// <summary>
    /// The scale_grad_by_freq parameter for the EmbeddingBag_from_pretrained module.
    /// </summary>
    [Description("The scale_grad_by_freq parameter for the EmbeddingBag_from_pretrained module")]
    public bool ScaleGradByFreq { get; set; } = false;

    /// <summary>
    /// The mode parameter for the EmbeddingBag_from_pretrained module.
    /// </summary>
    [Description("The mode parameter for the EmbeddingBag_from_pretrained module")]
    public EmbeddingBagMode Mode { get; set; } = EmbeddingBagMode.Mean;

    /// <summary>
    /// The sparse parameter for the EmbeddingBag_from_pretrained module.
    /// </summary>
    [Description("The sparse parameter for the EmbeddingBag_from_pretrained module")]
    public bool Sparse { get; set; } = false;

    /// <summary>
    /// The include_last_offset parameter for the EmbeddingBag_from_pretrained module.
    /// </summary>
    [Description("The include_last_offset parameter for the EmbeddingBag_from_pretrained module")]
    public bool IncludeLastOffset { get; set; } = false;

    /// <summary>
    /// The padding_index parameter for the EmbeddingBag_from_pretrained module.
    /// </summary>
    [Description("The padding_index parameter for the EmbeddingBag_from_pretrained module")]
    public long PaddingIndex { get; set; } = -1;

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
    /// Generates an observable sequence that creates a EmbeddingbagFromPretrainedModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(EmbeddingBag_from_pretrained(Embeddings, Freeze, MaxNorm, NormType, ScaleGradByFreq, Mode, Sparse, IncludeLastOffset, PaddingIndex, Device, Type));
    }
}
