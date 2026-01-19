using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Dropout;

/// <summary>
/// Represents an operator that creates a 1D dropout module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Dropout1d.html"/> for more information.
/// </remarks>
[Description("Creates a 1D dropout module.")]
public class Dropout1d
{
    /// <summary>
    /// The probability of an element to be zeroed.
    /// </summary>
    [Description("The probability of an element to be zeroed.")]
    public double Probability { get; set; } = 0.5D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place.")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Creates a Dropout1d module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Dropout1d> Process()
    {
        return Observable.Return(nn.Dropout1d(Probability, Inplace));
    }

    /// <summary>
    /// Creates a Dropout1d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Dropout1d> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => nn.Dropout1d(Probability, Inplace));
    }
}
