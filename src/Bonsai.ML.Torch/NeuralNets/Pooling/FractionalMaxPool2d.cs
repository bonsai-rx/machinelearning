using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Pooling;

/// <summary>
/// Represents an operator that creates a 2D fractional max pooling module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.FractionalMaxPool2d.html"/> for more information.
/// </remarks>
[Description("Creates a 2D fractional max pooling module.")]
public class FractionalMaxPool2d
{
    /// <summary>
    /// The size of the window to take a max over.
    /// </summary>
    [Description("The size of the window to take a max over.")]
    [TypeConverter(typeof(ValueTupleConverter<long, long>))]
    public (long, long) KernelSize { get; set; }

    /// <summary>
    /// The output size.
    /// </summary>
    [Description("The output size.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long>))]
    public (long, long)? OutputSize { get; set; } = null;

    /// <summary>
    /// Can be used to specify the output size as a ratio of the input size.
    /// </summary>
    [Description("Can be used to specify the output size as a ratio of the input size.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long>))]
    public (long, long)? OutputRatio { get; set; } = null;

    /// <summary>
    /// Creates a FractionalMaxPool2d module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(FractionalMaxPool2d(KernelSize, OutputSize, OutputRatio));
    }

    /// <summary>
    /// Creates a FractionalMaxPool2d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => FractionalMaxPool2d(KernelSize, OutputSize, OutputRatio));
    }
}
