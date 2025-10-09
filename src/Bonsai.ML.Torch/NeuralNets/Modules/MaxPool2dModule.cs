using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Modules;

/// <summary>
/// Creates a 2D max pooling layer.
/// </summary>
[Combinator]
[Description("Creates a 2D max pooling layer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MaxPool2dModule
{
    /// <summary>
    /// The kernelsize parameter for the MaxPool2d module.
    /// </summary>
    [Description("The kernelsize parameter for the MaxPool2d module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The stride parameter for the MaxPool2d module.
    /// </summary>
    [Description("The stride parameter for the MaxPool2d module")]
    public long? Stride { get; set; } = null;

    /// <summary>
    /// The padding parameter for the MaxPool2d module.
    /// </summary>
    [Description("The padding parameter for the MaxPool2d module")]
    public long? Padding { get; set; } = null;

    /// <summary>
    /// The dilation parameter for the MaxPool2d module.
    /// </summary>
    [Description("The dilation parameter for the MaxPool2d module")]
    public long? Dilation { get; set; } = null;

    /// <summary>
    /// The ceilmode parameter for the MaxPool2d module.
    /// </summary>
    [Description("The ceilmode parameter for the MaxPool2d module")]
    public bool CeilMode { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a MaxPool2dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(MaxPool2d(KernelSize, Stride, Padding, Dilation, CeilMode));
    }
}
