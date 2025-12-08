using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Convolution;

/// <summary>
/// Represents an operator that creates an unfold module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html"/> for more information.
/// </remarks>
[Description("Creates an unfold module.")]
public class Unfold
{
    /// <summary>
    /// The size of the sliding blocks.
    /// </summary>
    [Description("The size of the sliding blocks.")]
    [TypeConverter(typeof(ValueTupleConverter<long, long>))]
    public (long, long) KernelSize { get; set; }

    /// <summary>
    /// The stride of elements within the neighborhood.
    /// </summary>
    [Description("The stride of elements within the neighborhood.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long>))]
    public (long, long)? Dilation { get; set; } = null;

    /// <summary>
    /// The implicit zero-padding to be added on both sides of input.
    /// </summary>
    [Description("The implicit zero-padding to be added on both sides of input.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long>))]
    public (long, long)? Padding { get; set; } = null;

    /// <summary>
    /// The stride of the sliding blocks in the input tensor.
    /// </summary>
    [Description("The stride of the sliding blocks in the input tensor.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long>))]
    public (long, long)? Stride { get; set; } = null;

    /// <summary>
    /// Creates an Unfold module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Unfold(KernelSize, Dilation, Padding, Stride));
    }

    /// <summary>
    /// Creates an Unfold module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Unfold(KernelSize, Dilation, Padding, Stride));
    }
}
