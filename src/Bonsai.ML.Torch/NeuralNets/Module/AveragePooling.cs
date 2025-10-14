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
/// Creates a 1D adaptive average pooling layer.
/// </summary>
[Combinator]
[Description("Creates a 1D adaptive average pooling layer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class AveragePooling
{
    /// <summary>
    /// The number of dimensions for the AdaptiveAvgPool module.
    /// </summary>
    [Description("The number of dimensions for the AdaptiveAvgPool module")]
    public Dimensions Dimensions { get; set; } = Dimensions.One;

    /// <summary>
    /// The outputsize parameter for the AdaptiveAvgPool2d module.
    /// </summary>
    [Description("The outputsize parameter for the AdaptiveAvgPool2d module")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] OutputSize { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a AdaptiveAvgPool1dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Dimensions switch
        {
            Dimensions.One => Observable.Return(AvgPool1d(OutputSize[0])),
            Dimensions.Two => Observable.Return(AvgPool2d(OutputSize)),
            Dimensions.Three => Observable.Return(AvgPool3d(OutputSize)),
            _ => throw new InvalidOperationException("The specified number of dimensions is not supported."),
        };
    }
}
