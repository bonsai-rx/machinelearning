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
/// Creates a 3D adaptive average pooling layer.
/// </summary>
[Combinator]
[Description("Creates a 3D adaptive average pooling layer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class AdaptiveAvgPool3dModule
{
    /// <summary>
    /// The outputsize parameter for the AdaptiveAvgPool3d module.
    /// </summary>
    [Description("The outputsize parameter for the AdaptiveAvgPool3d module")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] OutputSize { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a AdaptiveAvgPool3dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(AdaptiveAvgPool3d(OutputSize));
    }
}
