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
/// Creates a Softshrink module.
/// </summary>
[Combinator]
[Description("Creates a Softshrink module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SoftShrinkage
{
    /// <summary>
    /// The lambda parameter for the Softshrink module.
    /// </summary>
    [Description("The lambda parameter for the Softshrink module")]
    public double Lambda { get; set; } = 0.5D;

    /// <summary>
    /// Generates an observable sequence that creates a SoftshrinkModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Softshrink(Lambda));
    }
}
