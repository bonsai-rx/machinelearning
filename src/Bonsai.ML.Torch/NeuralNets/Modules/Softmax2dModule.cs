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
/// Creates a Softmax2d module.
/// </summary>
[Combinator]
[Description("Creates a Softmax2d module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Softmax2dModule
{
    /// <summary>
    /// Generates an observable sequence that creates a Softmax2dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Softmax2d());
    }
}
