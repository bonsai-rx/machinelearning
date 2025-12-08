using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Distance;

/// <summary>
/// Represents an operator that creates a pairwise distance module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.PairwiseDistance.html"/> for more information.
/// </remarks>
[Description("Creates a pairwise distance module.")]
public class PairwiseDistance
{
    /// <summary>
    /// The norm degree which can be positive or negative.
    /// </summary>
    [Description("The norm degree which can be positive or negative.")]
    public double P { get; set; } = 2D;

    /// <summary>
    /// The value added to the denominator to avoid division by zero.
    /// </summary>
    [Description("The value added to the denominator to avoid division by zero.")]
    public double Eps { get; set; } = 1E-06D;

    /// <summary>
    /// Determines whether or not to keep the vector dimension.
    /// </summary>
    [Description("Determines whether or not to keep the vector dimension.")]
    public bool KeepDim { get; set; } = false;

    /// <summary>
    /// Creates a PairwiseDistance module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(PairwiseDistance(P, Eps, KeepDim));
    }

    /// <summary>
    /// Creates a PairwiseDistance module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => PairwiseDistance(P, Eps, KeepDim));
    }
}
