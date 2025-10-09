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
/// Creates a 1D adaptive max pooling layer.
/// </summary>
[Combinator]
[Description("Creates a 1D adaptive max pooling layer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class AdaptiveMaxPool1dModule
{
    /// <summary>
    /// The outputsize parameter for the AdaptiveMaxPool1d module.
    /// </summary>
    [Description("The outputsize parameter for the AdaptiveMaxPool1d module")]
    public long OutputSize { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a AdaptiveMaxPool1dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(AdaptiveMaxPool1d(OutputSize));
    }
}
