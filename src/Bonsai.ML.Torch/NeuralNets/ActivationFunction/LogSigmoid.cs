using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a log sigmoid module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html"/> for more information.
/// </remarks>
[Description("Creates a log sigmoid module.")]
public class LogSigmoid
{
    /// <summary>
    /// Creates a LogSigmoid module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.LogSigmoid> Process()
    {
        return Observable.Return(LogSigmoid());
    }

    /// <summary>
    /// Creates a LogSigmoid module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.LogSigmoid> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => LogSigmoid());
    }
}
