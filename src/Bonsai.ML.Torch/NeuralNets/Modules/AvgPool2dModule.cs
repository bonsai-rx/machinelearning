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
/// Creates a 2D average pooling layer module.
/// </summary>
[Combinator]
[Description("Creates a 2D average pooling layer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class AvgPool2dModule
{
    /// <summary>
    /// The kernel_size parameter for the AvgPool2d module.
    /// </summary>
    [Description("The kernel_size parameter for the AvgPool2d module")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] KernelSize { get; set; }

    /// <summary>
    /// The strides parameter for the AvgPool2d module.
    /// </summary>
    [Description("The strides parameter for the AvgPool2d module")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] Strides { get; set; } = null;

    /// <summary>
    /// The padding parameter for the AvgPool2d module.
    /// </summary>
    [Description("The padding parameter for the AvgPool2d module")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] Padding { get; set; } = null;

    /// <summary>
    /// The ceil_mode parameter for the AvgPool2d module.
    /// </summary>
    [Description("The ceil_mode parameter for the AvgPool2d module")]
    public bool CeilMode { get; set; } = false;

    /// <summary>
    /// The count_include_pad parameter for the AvgPool2d module.
    /// </summary>
    [Description("The count_include_pad parameter for the AvgPool2d module")]
    public bool CountIncludePad { get; set; } = true;

    /// <summary>
    /// The divisor_override parameter for the AvgPool2d module.
    /// </summary>
    [Description("The divisor_override parameter for the AvgPool2d module")]
    public long? DivisorOverride { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a AvgPool2d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(AvgPool2d(KernelSize, Strides, Padding, CeilMode, CountIncludePad, DivisorOverride));
    }
}
