using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Pooling;

/// <summary>
/// Represents an operator that creates a 1D adaptive max pooling layer.
/// </summary>
[Description("Creates a 1D adaptive max pooling layer.")]
public class AdaptiveMaxPool1d
{
    /// <summary>
    /// The output size.
    /// </summary>
    [Description("The output size")]
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
