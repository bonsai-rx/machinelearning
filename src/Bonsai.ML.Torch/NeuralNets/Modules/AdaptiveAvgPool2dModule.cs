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
/// Creates a 2D adaptive average pooling layer module.
/// </summary>
[Combinator]
[Description("Creates a 2D adaptive average pooling layer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class AdaptiveAvgPool2dModule
{
    /// <summary>
    /// The outputsize parameter for the AdaptiveAvgPool2d module.
    /// </summary>
    [Description("The outputsize parameter for the AdaptiveAvgPool2d module")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] OutputSize { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a AdaptiveAvgPool2d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(AdaptiveAvgPool2d(OutputSize));
    }
}
