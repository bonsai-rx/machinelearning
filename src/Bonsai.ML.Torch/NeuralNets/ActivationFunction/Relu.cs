using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a rectified linear unit (ReLU) activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html"/> for more information.
/// </remarks>
[Description("Creates a rectified linear unit (ReLU) activation function.")]
[DisplayName("ReLU")]
public class Relu
{
    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place.")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Creates a rectified linear unit (ReLU) module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.ReLU> Process()
    {
        return Observable.Return(ReLU(Inplace));
    }

    /// <summary>
    /// Creates a rectified linear unit (ReLU) module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.ReLU> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => ReLU(Inplace));
    }
}
