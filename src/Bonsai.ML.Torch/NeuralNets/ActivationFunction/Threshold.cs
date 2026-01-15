using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a threshold activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Threshold.html"/> for more information.
/// </remarks>
[Description("Represents an operator that creates a threshold activation function.")]
public class Threshold
{
    /// <summary>
    /// The threshold value.
    /// </summary>
    [Description("The threshold value.")]
    public double ThresholdValue { get; set; }

    /// <summary>
    /// The value used to replace values below the threshold.
    /// </summary>
    [Description("The value used to replace values below the threshold.")]
    public double FillValue { get; set; }

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place.")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Creates a Threshold module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Threshold(ThresholdValue, FillValue, Inplace));
    }

    /// <summary>
    /// Creates a Threshold module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Threshold(ThresholdValue, FillValue, Inplace));
    }
}
