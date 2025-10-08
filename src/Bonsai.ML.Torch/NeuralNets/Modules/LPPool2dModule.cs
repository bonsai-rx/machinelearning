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
/// Creates a LPPool2d module module.
/// </summary>
[Combinator]
[Description("Creates a LPPool2d module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LPPool2dModule
{
    /// <summary>
    /// The norm_type parameter for the LPPool2d module.
    /// </summary>
    [Description("The norm_type parameter for the LPPool2d module")]
    public double NormType { get; set; }

    /// <summary>
    /// The kernel_size parameter for the LPPool2d module.
    /// </summary>
    [Description("The kernel_size parameter for the LPPool2d module")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] KernelSize { get; set; }

    /// <summary>
    /// The strides parameter for the LPPool2d module.
    /// </summary>
    [Description("The strides parameter for the LPPool2d module")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] Strides { get; set; } = null;

    /// <summary>
    /// The ceil_mode parameter for the LPPool2d module.
    /// </summary>
    [Description("The ceil_mode parameter for the LPPool2d module")]
    public bool CeilMode { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a LPPool2d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(LPPool2d(NormType, KernelSize, Strides, CeilMode));
    }
}
