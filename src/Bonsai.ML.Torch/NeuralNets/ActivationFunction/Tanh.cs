using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a hyperbolic tangent (tanh) activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html"/> for more information.
/// </remarks>
[Description("Creates a hyperbolic tangent (tanh) activation function.")]
public class Tanh
{
    /// <summary>
    /// Creates a Tanh module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Tanh> Process()
    {
        return Observable.Return(Tanh());
    }

    /// <summary>
    /// Creates a Tanh module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Tanh> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Tanh());
    }
}
