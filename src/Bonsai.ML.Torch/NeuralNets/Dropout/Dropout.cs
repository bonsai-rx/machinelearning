using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Dropout;

/// <summary>
/// Represents an operator that creates a dropout module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"/> for more information.
/// </remarks>
[Description("Creates a dropout module.")]
public class Dropout
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
    /// Creates a Dropout module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Dropout> Process()
    {
        return Observable.Return(nn.Dropout(Probability, Inplace));
    }

    /// <summary>
    /// Creates a Dropout module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Dropout> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => nn.Dropout(Probability, Inplace));
    }
}
