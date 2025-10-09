using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Losses;

/// <summary>
/// Creates a NLLLoss module.
/// </summary>
[Combinator]
[Description("Creates a NLLLoss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class NLLLoss
{
    /// <summary>
    /// The weight parameter for the NLLLoss module.
    /// </summary>
    [Description("The weight parameter for the NLLLoss module")]
    public torch.Tensor Weight { get; set; } = null;

    /// <summary>
    /// The reduction parameter for the NLLLoss module.
    /// </summary>
    [Description("The reduction parameter for the NLLLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a NLLLoss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(NLLLoss(Weight, Reduction));
    }
}
