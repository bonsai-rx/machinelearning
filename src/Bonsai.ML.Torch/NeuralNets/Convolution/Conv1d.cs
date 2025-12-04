using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Convolution;

/// <summary>
/// Represents an operator that creates a 1D convolution module.
/// </summary>
[Description("Creates a 1D convolution module.")]
public class Conv1d
{
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
    /// The desired device of returned tensor.
    /// </summary>
    [XmlIgnore]
    [Description("The desired device of returned tensor")]
    public Device Device { get; set; } = null;

    /// <summary>
    /// The desired data type of returned tensor.
    /// </summary>
    [Description("The desired data type of returned tensor")]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Creates a Conv1d module.
    /// </summary>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Conv1d(InChannels, OutChannels, KernelSize, Stride, Padding, Dilation, PaddingMode, Groups, Bias, Device, Type));
    }

    /// <summary>
    /// Creates a Conv1d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Conv1d(InChannels, OutChannels, KernelSize, Stride, Padding, Dilation, PaddingMode, Groups, Bias, Device, Type));
    }
}
