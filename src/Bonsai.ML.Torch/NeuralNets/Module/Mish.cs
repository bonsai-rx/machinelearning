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
/// Creates a Mish module.
/// </summary>
[Combinator]
[Description("Creates a Mish module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Mish
{
    /// <summary>
    /// Generates an observable sequence that creates a MishModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Mish());
    }
}
