using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a mish activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Mish.html"/> for more information.
/// </remarks>
[Description("Creates a mish activation function.")]
public class Mish
{
    /// <summary>
    /// Creates a Mish module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Mish> Process()
    {
        return Observable.Return(Mish());
    }

    /// <summary>
    /// Creates a Mish module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Mish> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Mish());
    }
}
