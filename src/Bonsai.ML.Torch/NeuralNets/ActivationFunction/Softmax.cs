using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a softmax activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html"/> for more information.
/// </remarks>
[Description("Creates a softmax activation function.")]
public class Softmax
{
    /// <summary>
    /// The dimension along which Softmax will be computed.
    /// </summary>
    [Description("The dimension along which Softmax will be computed.")]
    public long Dim { get; set; }

    /// <summary>
    /// Creates a Softmax module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Softmax(Dim));
    }

    /// <summary>
    /// Creates a Softmax module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Softmax(Dim));
    }
}
