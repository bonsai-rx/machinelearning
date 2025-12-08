using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Pooling;

/// <summary>
/// Represents an operator that creates a 2D average pooling module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html"/> for more information.
/// </remarks>
[Description("Creates a 2D average pooling module.")]
public class AvgPool2d
{
    /// <summary>
    /// The size of the window.
    /// </summary>
    [Description("The size of the window.")]
    [TypeConverter(typeof(ValueTupleConverter<long, long>))]
    public (long, long) KernelSize { get; set; }

    /// <summary>
    /// The stride of the window.
    /// </summary>
    [Description("The stride of the window.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long>))]
    public (long, long)? Stride { get; set; } = null;

    /// <summary>
    /// The implicit zero padding to be added on both sides.
    /// </summary>
    [Description("The implicit zero padding to be added on both sides.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long>))]
    public (long, long)? Padding { get; set; } = null;

    /// <summary>
    /// If set to true, will use ceil instead of floor to compute the output shape.
    /// </summary>
    [Description("If set to true, will use ceil instead of floor to compute the output shape.")]
    public bool CeilMode { get; set; } = false;

    /// <summary>
    /// If set to true, will include the zero-padding in the averaging calculation.
    /// </summary>
    [Description("If set to true, will include the zero-padding in the averaging calculation.")]
    public bool CountIncludePad { get; set; } = true;

    /// <summary>
    /// If specified, it will be used as divisor, otherwise size of the pooling region will be used.
    /// </summary>
    [Description("If specified, it will be used as divisor, otherwise size of the pooling region will be used.")]
    public long? DivisorOverride { get; set; } = null;

    /// <summary>
    /// Creates an AvgPool2d module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(AvgPool2d(KernelSize, Stride, Padding, CeilMode, CountIncludePad, DivisorOverride));
    }

    /// <summary>
    /// Creates an AvgPool2d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => AvgPool2d(KernelSize, Stride, Padding, CeilMode, CountIncludePad, DivisorOverride));
    }
}
