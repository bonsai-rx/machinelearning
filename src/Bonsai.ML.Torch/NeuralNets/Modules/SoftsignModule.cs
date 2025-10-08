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
/// Creates a Softsign module module.
/// </summary>
[Combinator]
[Description("Creates a Softsign module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SoftsignModule
{
    /// <summary>
    /// Generates an observable sequence that creates a Softsign module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Softsign());
    }
}
