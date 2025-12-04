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
/// Represents an operator that creates a 1D adaptive average pooling layer.
/// </summary>
[Description("Creates a 1D adaptive average pooling layer.")]
public class AdaptiveAvgPool1d
{
    /// <summary>
    /// The output size.
    /// </summary>
    [Description("The output size")]
    public long OutputSize { get; set; }

    /// <summary>
    /// Creates an AdaptiveAvgPool1d module.
    /// </summary>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(AdaptiveAvgPool1d(OutputSize));
    }

    /// <summary>
    /// Creates an AdaptiveAvgPool1d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => AdaptiveAvgPool1d(OutputSize));
    }
}
