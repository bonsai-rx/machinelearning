using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Pooling;

/// <summary>
/// Represents an operator that creates a 3D max pooling layer.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html"/> for more information.
/// </remarks>
[Description("Creates a 3D max pooling layer.")]
public class MaxPool3d
{
    /// <summary>
    /// The size of the window to take a max over.
    /// </summary>
    [Description("The size of the window to take a max over.")]
    [TypeConverter(typeof(ValueTupleConverter<long, long, long>))]
    public (long, long, long) KernelSize { get; set; }

    /// <summary>
    /// The stride of the window.
    /// </summary>
    [Description("The stride of the window.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long, long>))]
    public (long, long, long)? Stride { get; set; } = null;

    /// <summary>
    /// The implicit negative infinity padding to be added on all three sides.
    /// </summary>
    [Description("The implicit negative infinity padding to be added on all three sides.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long, long>))]
    public (long, long, long)? Padding { get; set; } = null;

    /// <summary>
    /// The spacing between kernel elements.
    /// </summary>
    [Description("The spacing between kernel elements.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long, long>))]
    public (long, long, long)? Dilation { get; set; } = null;

    /// <summary>
    /// If set to true, will use ceil instead of floor to compute the output shape.
    /// </summary>
    [Description("If set to true, will use ceil instead of floor to compute the output shape.")]
    public bool CeilMode { get; set; } = false;

    /// <summary>
    /// Creates a MaxPool3d module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.MaxPool3d> Process()
    {
        return Observable.Return(MaxPool3d(KernelSize, Stride, Padding, Dilation, CeilMode));
    }

    /// <summary>
    /// Creates a MaxPool3d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.MaxPool3d> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => MaxPool3d(KernelSize, Stride, Padding, Dilation, CeilMode));
    }
}
