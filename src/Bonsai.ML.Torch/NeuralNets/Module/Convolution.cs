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
/// Creates a 1D convolution layer.
/// </summary>
[Combinator]
[Description("Creates a 1D convolution layer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Convolution
{
    /// <summary>
    /// The number of dimensions for the AdaptiveAvgPool module.
    /// </summary>
    [Description("The number of dimensions for the AdaptiveAvgPool module")]
    public Dimensions Dimensions { get; set; } = Dimensions.One;

    /// <summary>
    /// The in_channels parameter for the Conv1d module.
    /// </summary>
    [Description("The in_channels parameter for the Conv1d module")]
    public long InChannels { get; set; }

    /// <summary>
    /// The out_channels parameter for the Conv1d module.
    /// </summary>
    [Description("The out_channels parameter for the Conv1d module")]
    public long OutChannels { get; set; }

    /// <summary>
    /// The kernelsize parameter for the Conv1d module.
    /// </summary>
    [Description("The kernelsize parameter for the Conv1d module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The stride parameter for the Conv1d module.
    /// </summary>
    [Description("The stride parameter for the Conv1d module")]
    public long Stride { get; set; } = 1;

    /// <summary>
    /// The padding parameter for the Conv1d module.
    /// </summary>
    [Description("The padding parameter for the Conv1d module")]
    public long Padding { get; set; } = 0;

    /// <summary>
    /// The output_padding parameter for the Conv1d module.
    /// </summary>
    [Description("The output_padding parameter for the ConvTransposed1d module")]
    public long OutputPadding { get; set; } = 0;

    /// <summary>
    /// The dilation parameter for the Conv1d module.
    /// </summary>
    [Description("The dilation parameter for the Conv1d module")]
    public long Dilation { get; set; } = 1;

    /// <summary>
    /// The padding_mode parameter for the Conv1d module.
    /// </summary>
    [Description("The padding_mode parameter for the Conv1d module")]
    public PaddingModes PaddingMode { get; set; } = PaddingModes.Zeros;

    /// <summary>
    /// The groups parameter for the Conv1d module.
    /// </summary>
    [Description("The groups parameter for the Conv1d module")]
    public long Groups { get; set; } = 1;

    /// <summary>
    /// If true, adds a learnable bias to the output.
    /// </summary>
    [Description("If true, adds a learnable bias to the output")]
    public bool Bias { get; set; } = true;

    /// <summary>
    /// If true, creates a transposed convolution layer.
    /// </summary>
    [Description("If true, creates a transposed convolution layer")]
    public bool Transposed { get; set; } = false;

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
    /// Generates an observable sequence that creates a Conv1dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Dimensions switch
        {
            Dimensions.One => Transposed
                ? Observable.Return(ConvTranspose1d(InChannels, OutChannels, KernelSize, Stride, Padding, 0, Dilation, PaddingMode, Groups, Bias, Device, Type))
                : Observable.Return(Conv1d(InChannels, OutChannels, KernelSize, Stride, Padding, Dilation, PaddingMode, Groups, Bias, Device, Type)),
            Dimensions.Two => Transposed
                ? Observable.Return(ConvTranspose2d(InChannels, OutChannels, KernelSize, Stride, Padding, OutputPadding, Dilation, PaddingMode, Groups, Bias, Device, Type))
                : Observable.Return(Conv2d(InChannels, OutChannels, KernelSize, Stride, Padding, Dilation, PaddingMode, Groups, Bias, Device, Type)),
            Dimensions.Three => Transposed
                ? Observable.Return(ConvTranspose3d(InChannels, OutChannels, KernelSize, Stride, Padding, OutputPadding, Dilation, PaddingMode, Groups, Bias, Device, Type))
                : Observable.Return(Conv3d(InChannels, OutChannels, KernelSize, Stride, Padding, Dilation, PaddingMode, Groups, Bias, Device, Type)),
            _ => throw new InvalidOperationException("The specified number of dimensions is not supported."),
        };
    }
}
