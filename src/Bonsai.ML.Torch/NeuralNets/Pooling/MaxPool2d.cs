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
/// Represents an operator that creates a 2D max pooling layer.
/// </summary>
[Description("Creates a 2D max pooling layer.")]
public class MaxPool2d
{
    /// <summary>
    /// The kernel size.
    /// </summary>
    [Description("The kernel size.")]
    public long[] KernelSize { get; set; }

    /// <summary>
    /// The stride.
    /// </summary>
    [Description("The stride.")]
    public long[] Stride { get; set; } = null;

    /// <summary>
    /// The padding.
    /// </summary>
    [Description("The padding.")]
    public long[] Padding { get; set; } = null;

    /// <summary>
    /// The dilation.
    /// </summary>
    [Description("The dilation.")]
    public long[] Dilation { get; set; } = null;

    /// <summary>
    /// The ceiling mode.
    /// </summary>
    [Description("The ceiling mode.")]
    public bool CeilMode { get; set; } = false;

    /// <summary>
    /// Creates an MaxPool2d module.
    /// </summary>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(MaxPool2d(KernelSize, Stride, Padding, Dilation, CeilMode));
    }

    /// <summary>
    /// Creates an MaxPool2d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => MaxPool2d(KernelSize, Stride, Padding, Dilation, CeilMode));
    }
}
