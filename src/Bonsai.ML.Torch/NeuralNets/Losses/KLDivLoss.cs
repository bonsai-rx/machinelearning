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
/// Creates a KLDivLoss module.
/// </summary>
[Combinator]
[Description("Creates a KLDivLoss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class KLDivLoss
{
    /// <summary>
    /// The log_target parameter for the KLDivLoss module.
    /// </summary>
    [Description("The log_target parameter for the KLDivLoss module")]
    public bool LogTarget { get; set; } = true;

    /// <summary>
    /// The reduction parameter for the KLDivLoss module.
    /// </summary>
    [Description("The reduction parameter for the KLDivLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a KLDivLoss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(KLDivLoss(LogTarget, Reduction));
    }
}
