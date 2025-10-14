using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Bonsai.Reactive;

namespace Bonsai.ML.Torch.NeuralNets.Module;

/// <summary>
/// Creates a LPPool1d module.
/// </summary>
[Combinator]
[Description("Creates a LPPool1d module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MaxPooling
{
    /// <summary>
    /// The number of dimensions for the AdaptiveAvgPool module.
    /// </summary>
    [Description("The number of dimensions for the AdaptiveAvgPool module")]
    public Dimensions Dimensions { get; set; } = Dimensions.One;
    
    /// <summary>
    /// The kernelsize parameter for the MaxPool3d module.
    /// </summary>
    [Description("The kernelsize parameter for the MaxPool3d module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The stride parameter for the MaxPool3d module.
    /// </summary>
    [Description("The stride parameter for the MaxPool3d module")]
    public long? Stride { get; set; } = null;

    /// <summary>
    /// The padding parameter for the MaxPool3d module.
    /// </summary>
    [Description("The padding parameter for the MaxPool3d module")]
    public long? Padding { get; set; } = null;

    /// <summary>
    /// The dilation parameter for the MaxPool3d module.
    /// </summary>
    [Description("The dilation parameter for the MaxPool3d module")]
    public long? Dilation { get; set; } = null;

    /// <summary>
    /// The ceilmode parameter for the MaxPool3d module.
    /// </summary>
    [Description("The ceilmode parameter for the MaxPool3d module")]
    public bool CeilMode { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a LPPool1dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Dimensions switch
        {
            Dimensions.One => Observable.Return(MaxPool1d(KernelSize, Stride, Padding, Dilation, CeilMode)),
            Dimensions.Two => Observable.Return(MaxPool2d(KernelSize, Stride, Padding, Dilation, CeilMode)),
            Dimensions.Three => Observable.Return(MaxPool3d(KernelSize, Stride, Padding, Dilation, CeilMode)),
            _ => throw new InvalidOperationException("The specified number of dimensions is not supported."),
        };
    }
}
