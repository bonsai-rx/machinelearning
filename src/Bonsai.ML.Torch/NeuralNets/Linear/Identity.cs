using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Linear;

/// <summary>
/// Represents an operator that creates an Identity module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Identity.html"/> for more information.
/// </remarks>
[Description("Creates an Identity module.")]
public class Identity
{
    /// <summary>
    /// Creates an Identity module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(nn.Identity());
    }

    /// <summary>
    /// Creates an Identity module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => nn.Identity());
    }
}
