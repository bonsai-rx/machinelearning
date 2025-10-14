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
/// Creates a L1Loss module.
/// </summary>
[Combinator]
[Description("Creates a L1Loss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MeanAbsoluteError
{
    /// <summary>
    /// The reduction parameter for the L1Loss module.
    /// </summary>
    [Description("The reduction parameter for the L1Loss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a L1Loss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(L1Loss(Reduction));
    }
}
