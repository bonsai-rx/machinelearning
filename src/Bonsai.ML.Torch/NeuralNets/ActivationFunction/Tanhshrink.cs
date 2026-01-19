using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a tanhshrink activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Tanhshrink.html"/> for more information.
/// </remarks>
[Description("Creates a tanhshrink activation function.")]
public class Tanhshrink
{
    /// <summary>
    /// Creates a Tanhshrink module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Tanhshrink> Process()
    {
        return Observable.Return(Tanhshrink());
    }

    /// <summary>
    /// Creates a Tanhshrink module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Tanhshrink> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Tanhshrink());
    }
}
