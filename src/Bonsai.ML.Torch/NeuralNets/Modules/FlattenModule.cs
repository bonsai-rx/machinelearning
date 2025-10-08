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
/// Creates a Flatten module module.
/// </summary>
[Combinator]
[Description("Creates a Flatten module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class FlattenModule
{
    /// <summary>
    /// The startdim parameter for the Flatten module.
    /// </summary>
    [Description("The startdim parameter for the Flatten module")]
    public long StartDim { get; set; } = 1;

    /// <summary>
    /// The enddim parameter for the Flatten module.
    /// </summary>
    [Description("The enddim parameter for the Flatten module")]
    public long EndDim { get; set; } = -1;

    /// <summary>
    /// Generates an observable sequence that creates a Flatten module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Flatten(StartDim, EndDim));
    }
}
