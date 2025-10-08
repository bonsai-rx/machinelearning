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
/// Creates a Hardswish module module.
/// </summary>
[Combinator]
[Description("Creates a Hardswish module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class HardswishModule
{
    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a Hardswish module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Hardswish(Inplace));
    }
}
