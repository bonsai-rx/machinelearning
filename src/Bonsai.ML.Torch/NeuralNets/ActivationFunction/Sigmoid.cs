using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a sigmoid activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html"/> for more information.
/// </remarks>
[Description("Creates a sigmoid activation function.")]
public class Sigmoid
{
    /// <summary>
    /// Creates a sigmoid module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Sigmoid> Process()
    {
        return Observable.Return(Sigmoid());
    }

    /// <summary>
    /// Creates a sigmoid module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Sigmoid> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Sigmoid());
    }
}
