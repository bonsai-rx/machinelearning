using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Represents an operator that creates a triplet margin loss (TripletMarginLoss) module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html"/> for more information.
/// </remarks>
[Description("Creates a triplet margin loss (TripletMarginLoss) module.")]
public class TripletMargin
{
    /// <summary>
    /// The margin parameter, a float value.
    /// </summary>
    [Description("The margin parameter, a float value.")]
    public double Margin { get; set; } = 1D;

    /// <summary>
    /// The p parameter, the norm degree for pairwise distance.
    /// </summary>
    [Description("The p parameter, the norm degree for pairwise distance.")]
    public long P { get; set; } = 2;

    /// <summary>
    /// The value added to the denominator for numerical stability.
    /// </summary>
    [Description("The value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-06D;

    /// <summary>
    /// Determines whether to use the swapped distance.
    /// </summary>
    [Description("Determines whether to use the swapped distance.")]
    public bool Swap { get; set; } = false;

    /// <summary>
    /// The reduction type to apply to the output.
    /// </summary>
    [Description("The reduction type to apply to the output.")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Creates a triplet margin loss (TripletMarginLoss) module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.TripletMarginLoss> Process()
    {
        return Observable.Return(TripletMarginLoss(Margin, P, Eps, Swap, Reduction));
    }

    /// <summary>
    /// Creates a triplet margin loss (TripletMarginLoss) module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.TripletMarginLoss> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => TripletMarginLoss(Margin, P, Eps, Swap, Reduction));
    }
}
