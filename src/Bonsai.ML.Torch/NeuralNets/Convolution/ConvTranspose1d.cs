using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Convolution;

/// <summary>
/// Represents an operator that creates a 1D transposed convolution module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html"/> for more information.
/// </remarks>
[Description("Creates a 1D transposed convolution module.")]
public class ConvTranspose1d
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
    public long KernelSize { get; set; }

    /// <summary>
    /// The stride of the convolution.
    /// </summary>
    [Description("The stride of the convolution.")]
    public long Stride { get; set; } = 1;

    /// <summary>
    /// The padding to add to both sides of the input.
    /// </summary>
    [Description("The padding to add to both sides of the input.")]
    public long Padding { get; set; } = 0;

    /// <summary>
    /// The additional size added to one side of the output shape.
    /// </summary>
    [Description("The additional size added to one side of the output shape.")]
    public long OutputPadding { get; set; } = 0;

    /// <summary>
    /// The spacing between kernel elements.
    /// </summary>
    [Description("The spacing between kernel elements.")]
    public long Dilation { get; set; } = 1;

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
    /// Creates a ConvTranspose1d module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.ConvTranspose1d> Process()
    {
        return Observable.Return(ConvTranspose1d(InChannels, OutChannels, KernelSize, Stride, Padding, OutputPadding, Dilation, PaddingMode, Groups, Bias, Device, Type));
    }

    /// <summary>
    /// Creates a ConvTranspose1d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.ConvTranspose1d> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => ConvTranspose1d(InChannels, OutChannels, KernelSize, Stride, Padding, OutputPadding, Dilation, PaddingMode, Groups, Bias, Device, Type));
    }
}
