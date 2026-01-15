using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a softshrink activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html"/> for more information.
/// </remarks>
[Description("Creates a softshrink activation function.")]
public class Softshrink
{
    /// <summary>
    /// The lambda value for the softshrink formula.
    /// </summary>
    [Description("The lambda value for the softshrink formula.")]
    public double Lambda { get; set; } = 0.5D;

    /// <summary>
    /// Creates a Softshrink module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Softshrink(Lambda));
    }

    /// <summary>
    /// Creates a Softshrink module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Softshrink(Lambda));
    }
}
