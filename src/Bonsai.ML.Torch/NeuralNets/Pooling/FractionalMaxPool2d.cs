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
/// Represents an operator that creates a 2D fractional max pooling layer.
/// </summary>
[Description("Creates a 2D fractional max pooling layer.")]
public class FractionalMaxPool2d
{
    /// <summary>
    /// The kernel size.
    /// </summary>
    [Description("The kernel size.")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The output size.
    /// </summary>
    [Description("The output size.")]
    public long? OutputSize { get; set; } = null;

    /// <summary>
    /// The output ratio.
    /// </summary>
    [Description("The output ratio.")]
    public long? OutputRatio { get; set; } = null;

    /// <summary>
    /// Creates a FractionalMaxPool2d module.
    /// </summary>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(FractionalMaxPool2d(KernelSize, OutputSize, OutputRatio));
    }

    /// <summary>
    /// Creates a FractionalMaxPool2d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => FractionalMaxPool2d(KernelSize, OutputSize, OutputRatio));
    }
}
