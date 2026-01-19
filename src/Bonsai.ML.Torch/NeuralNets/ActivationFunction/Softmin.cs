using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a softmin activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Softmin.html"/> for more information.
/// </remarks>
[Description("Creates a softmin activation function.")]
public class Softmin
{
    /// <summary>
    /// The dimension along which softmin will be computed.
    /// </summary>
    [Description("The dimension along which softmin will be computed.")]
    public long Dim { get; set; }

    /// <summary>
    /// Creates a Softmin module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Softmin> Process()
    {
        return Observable.Return(Softmin(Dim));
    }

    /// <summary>
    /// Creates a Softmin module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Softmin> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Softmin(Dim));
    }
}
