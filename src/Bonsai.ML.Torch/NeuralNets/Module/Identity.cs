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
/// Creates a Identity module.
/// </summary>
[Combinator]
[Description("Creates a Identity module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Identity
{
    /// <summary>
    /// Generates an observable sequence that creates a IdentityModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Identity());
    }
}
