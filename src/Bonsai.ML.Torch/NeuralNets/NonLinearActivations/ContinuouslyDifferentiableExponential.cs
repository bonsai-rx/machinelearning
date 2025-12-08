using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.NonLinearActivations;

/// <summary>
/// Represents an operator that creates a continuously differentiable exponential linear unit (CELU) activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.CELU.html"/> for more information.
/// </remarks>
[Description("Creates a continuously differentiable exponential linear unit (CELU) activation function.")]
public class ContinuouslyDifferentiableExponential
{
    /// <summary>
    /// The alpha value for the CELU activation function.
    /// </summary>
    [Description("The alpha value for the CELU activation function.")]
    public double Alpha { get; set; } = 1D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place.")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Creates a continuously differentiable exponential linear unit (CELU) module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(CELU(Alpha, Inplace));
    }

    /// <summary>
    /// Creates a continuously differentiable exponential linear unit (CELU) module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => CELU(Alpha, Inplace));
    }
}
