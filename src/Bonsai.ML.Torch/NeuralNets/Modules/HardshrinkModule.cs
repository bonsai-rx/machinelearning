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
/// Creates a Hardshrink module module.
/// </summary>
[Combinator]
[Description("Creates a Hardshrink module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class HardshrinkModule
{
    /// <summary>
    /// The lambda parameter for the Hardshrink module.
    /// </summary>
    [Description("The lambda parameter for the Hardshrink module")]
    public double Lambda { get; set; } = 0.5;

    /// <summary>
    /// Generates an observable sequence that creates a Hardshrink module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Hardshrink(Lambda));
    }
}
