using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a Hardsigmoid module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html"/> for more information.
/// </remarks>
[Description("Creates a Hardsigmoid module.")]
public class Hardsigmoid
{
    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Creates a Hardsigmoid module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Hardsigmoid(Inplace));
    }

    /// <summary>
    /// Creates a Hardsigmoid module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Hardsigmoid(Inplace));
    }
}
