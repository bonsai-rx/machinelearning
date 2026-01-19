using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a randomized leaky rectified linear unit (RReLU) module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.RReLU.html"/> for more information.
/// </remarks>
[Description("Creates a randomized leaky rectified linear unit (RReLU) module.")]
[DisplayName("RReLU")]
public class Rrelu
{
    /// <summary>
    /// The lower bound of the uniform distribution.
    /// </summary>
    [Description("The lower bound of the uniform distribution.")]
    public double Lower { get; set; } = 0.125D;

    /// <summary>
    /// The upper bound of the uniform distribution.
    /// </summary>
    [Description("The upper bound of the uniform distribution.")]
    public double Upper { get; set; } = 0.3333333333333333D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place.")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Creates a randomized leaky rectified linear unit (RReLU) module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.RReLU> Process()
    {
        return Observable.Return(RReLU(Lower, Upper, Inplace));
    }

    /// <summary>
    /// Creates a randomized leaky rectified linear unit (RReLU) module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.RReLU> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => RReLU(Lower, Upper, Inplace));
    }
}
