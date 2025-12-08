using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Normalization;

/// <summary>
/// Represents an operator that creates a layer normalization module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html"/> for more information.
/// </remarks>
[Description("Creates a layer normalization module.")]
public class LayerNorm
{
    /// <summary>
    /// The input shape from an expected input of size `[* x normalized_shape[0] x normalized_shape[1] x ... x normalized_shape[-1]]`.
    /// </summary>
    [Description("The input shape from an expected input of size `[* x normalized_shape[0] x normalized_shape[1] x ... x normalized_shape[-1]]`.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] NormalizedShape { get; set; }

    /// <summary>
    /// The value added to the denominator for numerical stability.
    /// </summary>
    [Description("The value added to the denominator for numerical stability.")]
    public double Eps { get; set; } = 1E-05D;

    /// <summary>
    /// If true, this module has learnable per-element affine parameters initialized to ones (for weights) and zeros (for biases).
    /// </summary>
    [Description("If true, this module has learnable per-element affine parameters initialized to ones (for weights) and zeros (for biases).")]
    public bool ElementwiseAffine { get; set; } = true;

    /// <summary>
    /// If true, adds a learnable bias to the output.
    /// </summary>
    [Description("If true, adds a learnable bias to the output.")]
    public bool Bias { get; set; } = true;

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
    /// Creates a LayerNorm module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(LayerNorm(NormalizedShape, Eps, ElementwiseAffine, Bias, Device, Type));
    }

    /// <summary>
    /// Creates a LayerNorm module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => LayerNorm(NormalizedShape, Eps, ElementwiseAffine, Bias, Device, Type));
    }
}
