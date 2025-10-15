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
/// Creates a Layer normalization.
/// </summary>
[Combinator]
[Description("Creates a Layer normalization.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LayerNormalization
{
    /// <summary>
    /// The normalized_shape parameter for the LayerNorm module.
    /// </summary>
    [Description("The normalized_shape parameter for the LayerNorm module")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] NormalizedShape { get; set; }

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-05D;

    /// <summary>
    /// The elementwise_affine parameter for the LayerNorm module.
    /// </summary>
    [Description("The elementwise_affine parameter for the LayerNorm module")]
    public bool ElementwiseAffine { get; set; } = true;

    /// <summary>
    /// If true, adds a learnable bias to the output.
    /// </summary>
    [Description("If true, adds a learnable bias to the output")]
    public bool Bias { get; set; } = true;

    /// <summary>
    /// The desired device of returned tensor.
    /// </summary>
    [XmlIgnore]
    [Description("The desired device of returned tensor")]
    public Device Device { get; set; } = null;

    /// <summary>
    /// The desired data type of returned tensor.
    /// </summary>
    [Description("The desired data type of returned tensor")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a LayerNormModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(LayerNorm(NormalizedShape, Eps, ElementwiseAffine, Bias, Device, Type));
    }
}
