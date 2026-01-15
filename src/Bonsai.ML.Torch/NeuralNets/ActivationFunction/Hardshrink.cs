using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a Hardshrink module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html"/> for more information.
/// </remarks>
[Description("Creates a Hardshrink module.")]
public class Hardshrink
{
    /// <summary>
    /// The lambda parameter for the Hardshrink function.
    /// </summary>
    [Description("The lambda parameter for the Hardshrink function")]
    public double Lambda { get; set; } = 0.5D;

    /// <summary>
    /// Creates a Hardshrink module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Hardshrink(Lambda));
    }

    /// <summary>
    /// Creates a Hardshrink module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Hardshrink(Lambda));
    }
}
