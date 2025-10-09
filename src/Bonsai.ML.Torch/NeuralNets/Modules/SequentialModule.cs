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
/// Creates a Sequential module.
/// </summary>
[Combinator]
[Description("Creates a Sequential module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SequentialModule
{
    /// <summary>
    /// Generates an observable sequence that creates a SequentialModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Sequential());
    }
}
