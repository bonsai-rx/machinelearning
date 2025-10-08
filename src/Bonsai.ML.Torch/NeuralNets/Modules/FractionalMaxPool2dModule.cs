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
/// Creates a FractionalMaxPool2d module module.
/// </summary>
[Combinator]
[Description("Creates a FractionalMaxPool2d module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class FractionalMaxPool2dModule
{
    /// <summary>
    /// The kernel_size parameter for the FractionalMaxPool2d module.
    /// </summary>
    [Description("The kernel_size parameter for the FractionalMaxPool2d module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The output_size parameter for the FractionalMaxPool2d module.
    /// </summary>
    [Description("The output_size parameter for the FractionalMaxPool2d module")]
    public long? OutputSize { get; set; } = null;

    /// <summary>
    /// The output_ratio parameter for the FractionalMaxPool2d module.
    /// </summary>
    [Description("The output_ratio parameter for the FractionalMaxPool2d module")]
    public double? OutputRatio { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a FractionalMaxPool2d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(FractionalMaxPool2d(KernelSize, OutputSize, OutputRatio));
    }
}
