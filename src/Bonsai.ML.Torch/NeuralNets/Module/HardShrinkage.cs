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
/// Creates a Hardshrink module.
/// </summary>
[Combinator]
[Description("Creates a Hardshrink module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class HardShrinkage
{
    /// <summary>
    /// The lambda parameter for the Hardshrink module.
    /// </summary>
    [Description("The lambda parameter for the Hardshrink module")]
    public double Lambda { get; set; } = 0.5D;

    /// <summary>
    /// Generates an observable sequence that creates a HardshrinkModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Hardshrink(Lambda));
    }
}
