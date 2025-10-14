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
/// Creates a dropout layer.
/// </summary>
[Combinator]
[Description("Creates a fractional max pooling layer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class FractionalMaxPooling
{
    /// <summary>
    /// The number of dimensions for the AdaptiveAvgPool module.
    /// </summary>
    [Description("The number of dimensions for the AdaptiveAvgPool module")]
    public Dimensions Dimensions { get; set; } = Dimensions.Two;

    /// <summary>
    /// The kernel_size parameter for the FractionalMaxPool3d module.
    /// </summary>
    [Description("The kernel_size parameter for the FractionalMaxPool3d module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The output_size parameter for the FractionalMaxPool3d module.
    /// </summary>
    [Description("The output_size parameter for the FractionalMaxPool3d module")]
    public long? OutputSize { get; set; } = null;

    /// <summary>
    /// The output_ratio parameter for the FractionalMaxPool3d module.
    /// </summary>
    [Description("The output_ratio parameter for the FractionalMaxPool3d module")]
    public double? OutputRatio { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a Dropout module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Dimensions switch
        {
            Dimensions.Two => Observable.Return(FractionalMaxPool2d(KernelSize, OutputSize, OutputRatio)),
            Dimensions.Three => Observable.Return(FractionalMaxPool3d(KernelSize, OutputSize, OutputRatio)),
            _ => throw new InvalidOperationException("The specified number of dimensions is not supported."),
        };
    }
}
