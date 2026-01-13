using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Convolution;

/// <summary>
/// Represents an operator that creates a 3D convolution module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html"/> for more information.
/// </remarks>
[Description("Creates a 3D convolution module.")]
public class Conv3d
{
    /// <summary>
    /// The number of input channels in the input tensor.
    /// </summary>
    [Description("The number of input channels in the input tensor.")]
    public long InChannels { get; set; }

    /// <summary>
    /// The number of output channels produced by the convolution.
    /// </summary>
    [Description("The number of output channels produced by the convolution.")]
    public long OutChannels { get; set; }

    /// <summary>
    /// The size of the convolution kernel.
    /// </summary>
    [Description("The size of the convolution kernel.")]
    [TypeConverter(typeof(ValueTupleConverter<long, long, long>))]
    public (long, long, long) KernelSize { get; set; }

    /// <summary>
    /// The stride of the convolution.
    /// </summary>
    [Description("The stride of the convolution.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long, long>))]
    public (long, long, long)? Stride { get; set; } = null;

    /// <summary>
    /// The padding to add to all six sides of the input.
    /// </summary>
    [Description("The padding to add to all six sides of the input.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long, long>))]
    public (long, long, long)? Padding { get; set; } = null;

    /// <summary>
    /// The spacing between kernel elements.
    /// </summary>
    [Description("The spacing between kernel elements.")]
    [TypeConverter(typeof(NullableValueTupleConverter<long, long, long>))]
    public (long, long, long)? Dilation { get; set; } = null;

    /// <summary>
    /// The mode of padding.
    /// </summary>
    [Description("The mode of padding.")]
    public PaddingModes PaddingMode { get; set; } = PaddingModes.Zeros;

    /// <summary>
    /// The number of blocked connections from input channels to output channels.
    /// </summary>
    [Description("The number of blocked connections from input channels to output channels.")]
    public long Groups { get; set; } = 1;

    /// <summary>
    /// If true, adds a learnable bias to the output.
    /// </summary>
    [Description("If true, adds a learnable bias to the output")]
    public bool Bias { get; set; } = true;

    /// <summary>
    /// The desired device of the returned tensor.
    /// </summary>
    [XmlIgnore]
    [Description("The desired device of the returned tensor")]
    public Device Device { get; set; } = null;

    /// <summary>
    /// The desired data type of the returned tensor.
    /// </summary>
    [Description("The desired data type of the returned tensor")]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Creates a Conv3d module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Conv3d(InChannels, OutChannels, KernelSize, Stride, Padding, Dilation, PaddingMode, Groups, Bias, Device, Type));
    }

    /// <summary>
    /// Creates a Conv3d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Conv3d(InChannels, OutChannels, KernelSize, Stride, Padding, Dilation, PaddingMode, Groups, Bias, Device, Type));
    }
}
