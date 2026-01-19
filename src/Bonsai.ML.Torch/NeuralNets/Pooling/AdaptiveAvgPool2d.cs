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
/// Represents an operator that creates a 2D adaptive average pooling module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html"/> for more information.
/// </remarks>
[Description("Creates a 2D adaptive average pooling module.")]
public class AdaptiveAvgPool2d
{
    /// <summary>
    /// The output size.
    /// </summary>
    [Description("The output size.")]
    [TypeConverter(typeof(ValueTupleConverter<long, long>))]
    public (long, long) OutputSize { get; set; }

    /// <summary>
    /// Creates an AdaptiveAvgPool2d module.
    /// </summary>
    public IObservable<TorchSharp.Modules.AdaptiveAvgPool2d> Process()
    {
        return Observable.Return(AdaptiveAvgPool2d(OutputSize));
    }

    /// <summary>
    /// Creates an AdaptiveAvgPool2d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.AdaptiveAvgPool2d> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => AdaptiveAvgPool2d(OutputSize));
    }
}
