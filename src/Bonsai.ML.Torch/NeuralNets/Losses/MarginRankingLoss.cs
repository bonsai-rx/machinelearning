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
/// Creates a MarginRankingLoss module.
/// </summary>
[Combinator]
[Description("Creates a MarginRankingLoss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MarginRankingLoss
{
    /// <summary>
    /// The margin parameter for the MarginRankingLoss module.
    /// </summary>
    [Description("The margin parameter for the MarginRankingLoss module")]
    public double Margin { get; set; } = 0D;

    /// <summary>
    /// The reduction parameter for the MarginRankingLoss module.
    /// </summary>
    [Description("The reduction parameter for the MarginRankingLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a MarginRankingLoss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(MarginRankingLoss(Margin, Reduction));
    }
}
