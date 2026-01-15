using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a 2d softmax activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Softmax2d.html"/> for more information.
/// </remarks>
[Description("Creates a 2d softmax activation function.")]
public class Softmax2d
{
    /// <summary>
    /// Creates a Softmax2d module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Softmax2d());
    }

    /// <summary>
    /// Creates a Softmax2d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Softmax2d());
    }
}
