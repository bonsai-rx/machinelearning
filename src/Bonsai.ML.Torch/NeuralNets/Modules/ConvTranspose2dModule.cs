using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Modules;

/// <summary>
/// Creates a 2D transposed convolution layer module.
/// </summary>
[Combinator]
[Description("Creates a 2D transposed convolution layer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ConvTranspose2dModule
{
    /// <summary>
    /// The in_channels parameter for the ConvTranspose2d module.
    /// </summary>
    [Description("The in_channels parameter for the ConvTranspose2d module")]
    public long InChannels { get; set; }

    /// <summary>
    /// The out_channels parameter for the ConvTranspose2d module.
    /// </summary>
    [Description("The out_channels parameter for the ConvTranspose2d module")]
    public long OutChannels { get; set; }

    /// <summary>
    /// The kernelsize parameter for the ConvTranspose2d module.
    /// </summary>
    [Description("The kernelsize parameter for the ConvTranspose2d module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The stride parameter for the ConvTranspose2d module.
    /// </summary>
    [Description("The stride parameter for the ConvTranspose2d module")]
    public long Stride { get; set; } = 1;

    /// <summary>
    /// The padding parameter for the ConvTranspose2d module.
    /// </summary>
    [Description("The padding parameter for the ConvTranspose2d module")]
    public long Padding { get; set; } = 0;

    /// <summary>
    /// The output_padding parameter for the ConvTranspose2d module.
    /// </summary>
    [Description("The output_padding parameter for the ConvTranspose2d module")]
    public long OutputPadding { get; set; } = 0;

    /// <summary>
    /// The dilation parameter for the ConvTranspose2d module.
    /// </summary>
    [Description("The dilation parameter for the ConvTranspose2d module")]
    public long Dilation { get; set; } = 1;

    /// <summary>
    /// The padding_mode parameter for the ConvTranspose2d module.
    /// </summary>
    [Description("The padding_mode parameter for the ConvTranspose2d module")]
    public PaddingModes PaddingMode { get; set; } = PaddingModes.Zeros;

    /// <summary>
    /// The groups parameter for the ConvTranspose2d module.
    /// </summary>
    [Description("The groups parameter for the ConvTranspose2d module")]
    public long Groups { get; set; } = 1;

    /// <summary>
    /// If true, adds a learnable bias to the output.
    /// </summary>
    [Description("If true, adds a learnable bias to the output")]
    public bool Bias { get; set; } = true;

    /// <summary>
    /// The desired device of returned tensor.
    /// </summary>
    [Description("The desired device of returned tensor")]
    [XmlIgnore]
    public Device Device { get; set; } = null;

    /// <summary>
    /// The desired data type of returned tensor.
    /// </summary>
    [Description("The desired data type of returned tensor")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a ConvTranspose2d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(ConvTranspose2d(InChannels, OutChannels, KernelSize, Stride, Padding, OutputPadding, Dilation, PaddingMode, Groups, Bias, Device, Type));
    }
}
