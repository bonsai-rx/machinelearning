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
/// Represents an operator that creates a 3D max unpooling module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool3d.html"/> for more information.
/// </remarks>
[Description("Creates a 3D max unpooling module.")]
public class MaxUnpool3d
{
    /// <summary>
    /// The size of the max pooling window.
    /// </summary>
    [Description("The size of the max pooling window.")]
    [TypeConverter(typeof(ValueTupleConverter<long, long, long>))]
    public (long, long, long) KernelSize { get; set; }

    /// <summary>
    /// The stride of the max pooling window.
    /// </summary>
    [Description("The stride of the max pooling window.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long, long>))]
    public (long, long, long)? Stride { get; set; } = null;

    /// <summary>
    /// The padding that was added to the input.
    /// </summary>
    [Description("The padding that was added to the input.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long, long>))]
    public (long, long, long)? Padding { get; set; } = null;

    /// <summary>
    /// Creates a MaxUnpool3d module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.MaxUnpool3d> Process()
    {
        return Observable.Return(MaxUnpool3d(KernelSize, Stride, Padding));
    }

    /// <summary>
    /// Creates a MaxUnpool3d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.MaxUnpool3d> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => MaxUnpool3d(KernelSize, Stride, Padding));
    }
}
