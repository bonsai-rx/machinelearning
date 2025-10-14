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
/// Creates a TripletMarginLoss module.
/// </summary>
[Combinator]
[Description("Creates a TripletMarginLoss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class TripletMargin
{
    /// <summary>
    /// The margin parameter for the TripletMarginLoss module.
    /// </summary>
    [Description("The margin parameter for the TripletMarginLoss module")]
    public double Margin { get; set; } = 1D;

    /// <summary>
    /// The p parameter for the TripletMarginLoss module.
    /// </summary>
    [Description("The p parameter for the TripletMarginLoss module")]
    public long P { get; set; } = 2;

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-06D;

    /// <summary>
    /// The swap parameter for the TripletMarginLoss module.
    /// </summary>
    [Description("The swap parameter for the TripletMarginLoss module")]
    public bool Swap { get; set; } = false;

    /// <summary>
    /// The reduction parameter for the TripletMarginLoss module.
    /// </summary>
    [Description("The reduction parameter for the TripletMarginLoss module")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Generates an observable sequence that creates a TripletMarginLoss.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(TripletMarginLoss(Margin, P, Eps, Swap, Reduction));
    }
}
