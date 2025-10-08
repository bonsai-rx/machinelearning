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
/// Creates a Sigmoid activation function module.
/// </summary>
[Combinator]
[Description("Creates a Sigmoid activation function module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SigmoidModule
{
    /// <summary>
    /// Generates an observable sequence that creates a Sigmoid module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Sigmoid());
    }
}
