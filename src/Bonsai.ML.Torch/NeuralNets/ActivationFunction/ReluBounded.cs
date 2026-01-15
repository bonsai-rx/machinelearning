using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a bounded rectified linear unit (ReLU6) activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html"/> for more information.
/// </remarks>
[Description("Creates a bounded rectified linear unit (ReLU6) activation function.")]
public class ReluBounded
{
    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place.")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Creates a bounded rectified linear unit (ReLU6) module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(ReLU6(Inplace));
    }

    /// <summary>
    /// Creates a bounded rectified linear unit (ReLU6) module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => ReLU6(Inplace));
    }
}
