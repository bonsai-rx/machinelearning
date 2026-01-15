using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a leaky rectified linear unit (LeakyReLU) activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html"/> for more information.
/// </remarks>
[Description("Creates a leaky rectified linear unit (LeakyReLU) activation function.")]
public class LeakyRelu
{
    /// <summary>
    /// The angle of the negative slope.
    /// </summary>
    [Description("The angle of the negative slope.")]
    public double NegativeSlope { get; set; } = 0.01D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place.")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Creates a leaky rectified linear unit (LeakyReLU) module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(LeakyReLU(NegativeSlope, Inplace));
    }

    /// <summary>
    /// Creates a leaky rectified linear unit (LeakyReLU) module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => LeakyReLU(NegativeSlope, Inplace));
    }
}
