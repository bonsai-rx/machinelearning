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
/// Creates a LPPool1d module.
/// </summary>
[Combinator]
[Description("Creates a LPPool1d module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PowerAveragePooling
{
    /// <summary>
    /// The number of dimensions for the AdaptiveAvgPool module.
    /// </summary>
    [Description("The number of dimensions for the AdaptiveAvgPool module")]
    public Dimensions Dimensions { get; set; } = Dimensions.One;

    /// <summary>
    /// The norm_type parameter for the LPPool1d module.
    /// </summary>
    [Description("The norm_type parameter for the LPPool1d module")]
    public double NormType { get; set; }

    /// <summary>
    /// The kernelsize parameter for the LPPool1d module.
    /// </summary>
    [Description("The kernelsize parameter for the LPPool1d module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The stride parameter for the LPPool1d module.
    /// </summary>
    [Description("The stride parameter for the LPPool1d module")]
    public long? Stride { get; set; } = null;

    /// <summary>
    /// The ceil_mode parameter for the LPPool1d module.
    /// </summary>
    [Description("The ceil_mode parameter for the LPPool1d module")]
    public bool CeilMode { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a LPPool1dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Dimensions switch
        {
            Dimensions.One => Observable.Return(LPPool1d(NormType, KernelSize, Stride, CeilMode)),
            Dimensions.Two => Observable.Return(LPPool2d(NormType, KernelSize, Stride, CeilMode)),
            _ => throw new InvalidOperationException("The specified number of dimensions is not supported."),
        };
    }
}
