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
/// Creates a Identity module module.
/// </summary>
[Combinator]
[Description("Creates a Identity module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class IdentityModule
{
    /// <summary>
    /// Generates an observable sequence that creates a Identity module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Identity());
    }
}
