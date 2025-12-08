using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Represents an operator that creates a Huber loss module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html"/> for more information.
/// </remarks>
[Description("Creates a Huber loss module.")]
public class Huber
{
    /// <summary>
    /// The threshold at which to change between delta-scaled L1 and L2 loss.
    /// </summary>
    [Description("The threshold at which to change between delta-scaled L1 and L2 loss.")]
    public double Delta { get; set; } = 1D;

    /// <summary>
    /// The reduction type to apply to the output.
    /// </summary>
    [Description("The reduction type to apply to the output.")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Creates a Huber loss module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(HuberLoss(Delta, Reduction));
    }

    /// <summary>
    /// Creates a Huber loss module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => HuberLoss(Delta, Reduction));
    }
}
