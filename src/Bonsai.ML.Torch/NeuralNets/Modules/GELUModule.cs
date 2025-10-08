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
/// Creates a GELU activation function module.
/// </summary>
[Combinator]
[Description("Creates a GELU activation function module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class GELUModule
{
    /// <summary>
    /// Generates an observable sequence that creates a GELU module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(GELU());
    }
}
