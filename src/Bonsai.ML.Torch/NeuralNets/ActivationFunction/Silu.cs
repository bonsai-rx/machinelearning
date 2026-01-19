using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a sigmoid weighted linear unit (SiLU) activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html"/> for more information.
/// </remarks>
[Description("Creates a sigmoid weighted linear unit (SiLU) activation function.")]
[DisplayName("SiLU")]
public class Silu
{
    /// <summary>
    /// Creates a sigmoid weighted linear unit (SiLU) module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.SiLU> Process()
    {
        return Observable.Return(SiLU());
    }

    /// <summary>
    /// Creates a sigmoid weighted linear unit (SiLU) module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.SiLU> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => SiLU());
    }
}
