using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Represents an operator that creates a smooth L1 loss (SmoothL1Loss) module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html"/> for more information.
/// </remarks>
[Description("Creates a smooth L1 loss (SmoothL1Loss) module.")]
public class SmoothL1
{
    /// <summary>
    /// The reduction type to apply to the output.
    /// </summary>
    [Description("The reduction type to apply to the output.")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// The threshold at which to change between L1 and L2 loss.
    /// </summary>
    [Description("The threshold at which to change between L1 and L2 loss.")]
    public double Beta { get; set; } = 1D;

    /// <summary>
    /// Creates a smooth L1 loss (SmoothL1Loss) module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(SmoothL1Loss(Reduction, Beta));
    }

    /// <summary>
    /// Creates a smooth L1 loss (SmoothL1Loss) module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => SmoothL1Loss(Reduction, Beta));
    }
}
