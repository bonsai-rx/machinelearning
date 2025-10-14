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
/// Creates a Flatten module.
/// </summary>
[Combinator]
[Description("Creates a Flatten module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Flatten
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
    /// Generates an observable sequence that creates a FlattenModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Flatten(StartDim, EndDim));
    }
}
