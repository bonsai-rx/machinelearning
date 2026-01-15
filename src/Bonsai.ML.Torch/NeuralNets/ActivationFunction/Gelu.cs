using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a gaussian error linear unit (GELU) activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.GELU.html"/> for more information.
/// </remarks>
[Description("Creates a gaussian error linear unit (GELU) activation function.")]
public class Gelu
{
    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place.")]
    public bool InPlace { get; set; } = false;

    /// <summary>
    /// Creates a gaussian error linear unit (GELU) module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(GELU(InPlace));
    }

    /// <summary>
    /// Creates a gaussian error linear unit (GELU) module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => GELU(InPlace));
    }
}
