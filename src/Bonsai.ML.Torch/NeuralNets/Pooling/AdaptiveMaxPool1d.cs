using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Pooling;

/// <summary>
/// Represents an operator that creates a 1D adaptive max pooling module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool1d.html"/> for more information.
/// </remarks>
[Description("Creates a 1D adaptive max pooling module.")]
public class AdaptiveMaxPool1d
{
    /// <summary>
    /// The output size.
    /// </summary>
    [Description("The output size.")]
    public long OutputSize { get; set; }

    /// <summary>
    /// Creates an AdaptiveMaxPool1d module.
    /// </summary>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(AdaptiveMaxPool1d(OutputSize));
    }

    /// <summary>
    /// Creates an AdaptiveMaxPool1d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => AdaptiveMaxPool1d(OutputSize));
    }
}
