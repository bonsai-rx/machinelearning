using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Creates a SmoothL1Loss module.
/// </summary>
[Combinator]
[Description("Creates a SmoothL1Loss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SmoothL1
{
    /// <summary>
    /// The reduction parameter for the SmoothL1Loss module.
    /// </summary>
    [Description("The reduction parameter for the SmoothL1Loss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// The beta parameter for the SmoothL1Loss module.
    /// </summary>
    [Description("The beta parameter for the SmoothL1Loss module")]
    public double Beta { get; set; } = 1D;

    /// <summary>
    /// Generates an observable sequence that creates a SmoothL1Loss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(SmoothL1Loss(Reduction, Beta));
    }
}
