using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Dropout;

/// <summary>
/// Represents an operator that creates a feature alpha dropout module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.FeatureAlphaDropout.html"/> for more information.
/// </remarks>
[Description("Creates a feature alpha dropout module.")]
public class FeatureAlphaDropout
{
    /// <summary>
    /// The probability of an element to be zeroed.
    /// </summary>
    [Description("The probability of an element to be zeroed.")]
    public double Probability { get; set; } = 0.5D;

    /// <summary>
    /// Creates a FeatureAlphaDropout module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.FeatureAlphaDropout> Process()
    {
        return Observable.Return(FeatureAlphaDropout(Probability));
    }

    /// <summary>
    /// Creates a FeatureAlphaDropout module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.FeatureAlphaDropout> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => FeatureAlphaDropout(Probability));
    }
}
