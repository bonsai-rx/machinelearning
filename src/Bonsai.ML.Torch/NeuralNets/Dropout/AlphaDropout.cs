using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Dropout;

/// <summary>
/// Represents an operator that creates an alpha dropout module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.AlphaDropout.html"/> for more information.
/// </remarks>
[Description("Creates an alpha dropout module.")]
public class AlphaDropout
{
    /// <summary>
    /// The probability of an element to be dropped.
    /// </summary>
    [Description("The probability of an element to be dropped.")]
    public double Probability { get; set; } = 0.5D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Creates an AlphaDropout module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(AlphaDropout(Probability, Inplace));
    }

    /// <summary>
    /// Creates an AlphaDropout module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => AlphaDropout(Probability, Inplace));
    }
}