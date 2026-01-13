using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.NonLinearActivations;

/// <summary>
/// Represents an operator that creates a tanh shrinkage activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Tanhshrink.html"/> for more information.
/// </remarks>
[Description("Creates a tanh shrinkage activation function.")]
public class TanhShrinkage
{
    /// <summary>
    /// Creates a Tanhshrink module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Tanhshrink());
    }

    /// <summary>
    /// Creates a Tanhshrink module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Tanhshrink());
    }
}
