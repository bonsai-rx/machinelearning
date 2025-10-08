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
/// Creates a Softshrink module module.
/// </summary>
[Combinator]
[Description("Creates a Softshrink module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SoftshrinkModule
{
    /// <summary>
    /// The lambda parameter for the Softshrink module.
    /// </summary>
    [Description("The lambda parameter for the Softshrink module")]
    public double Lambda { get; set; } = 0.5;

    /// <summary>
    /// Generates an observable sequence that creates a Softshrink module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Softshrink(Lambda));
    }
}
