using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Pooling;

/// <summary>
/// Creates a 1D convolution layer.
/// </summary>
[Combinator]
[Description("Creates a 1D convolution layer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MaxUnpoolModule
{
    /// <summary>
    /// The number of dimensions for the AdaptiveAvgPool module.
    /// </summary>
    [Description("The number of dimensions for the AdaptiveAvgPool module")]
    public Dimensions Dimensions { get; set; } = Dimensions.One;

    /// <summary>
    /// The kernelsize parameter for the MaxUnpool1d module.
    /// </summary>
    [Description("The kernelsize parameter for the MaxUnpool1d module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The stride parameter for the MaxUnpool1d module.
    /// </summary>
    [Description("The stride parameter for the MaxUnpool1d module")]
    public long? Stride { get; set; } = null;

    /// <summary>
    /// The padding parameter for the MaxUnpool1d module.
    /// </summary>
    [Description("The padding parameter for the MaxUnpool1d module")]
    public long? Padding { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a Conv1dModule module.
    /// </summary>
    public IObservable<Module<Tensor, Tensor, long[], Tensor>> Process()
    {
        return Dimensions switch
        {
            Dimensions.One => Observable.Return(MaxUnpool1d(KernelSize, Stride, Padding)),
            Dimensions.Two => Observable.Return(MaxUnpool2d(KernelSize, Stride, Padding)),
            Dimensions.Three => Observable.Return(MaxUnpool3d(KernelSize, Stride, Padding)),
            _ => throw new InvalidOperationException("The specified number of dimensions is not supported."),
        };
    }
}
