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
/// Represents an operator that creates a 2D max unpooling module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html"/> for more information.
/// </remarks>
[Description("Creates a 2D max unpooling module.")]
public class MaxUnpool2d
{
    /// <summary>
    /// The size of the max pooling window.
    /// </summary>
    [Description("The size of the max pooling window.")]
    [TypeConverter(typeof(ValueTupleConverter<long, long>))]
    public (long, long) KernelSize { get; set; }

    /// <summary>
    /// The stride of the max pooling window.
    /// </summary>
    [Description("The stride of the max pooling window.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long>))]
    public (long, long)? Stride { get; set; } = null;

    /// <summary>
    /// The padding that was added to the input.
    /// </summary>
    [Description("The padding that was added to the input.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long>))]
    public (long, long)? Padding { get; set; } = null;

    /// <summary>
    /// Creates a MaxUnpool2d module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, long[], Tensor>> Process()
    {
        return Observable.Return(MaxUnpool2d(KernelSize, Stride, Padding));
    }

    /// <summary>
    /// Creates a MaxUnpool2d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, long[], Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => MaxUnpool2d(KernelSize, Stride, Padding));
    }
}
