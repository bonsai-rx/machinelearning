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
/// Represents an operator that creates a 1D max pooling module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html"/> for more information.
/// </remarks>
[Description("Creates a 1D max pooling module.")]
public class MaxPool1d
{
    /// <summary>
    /// The size of the sliding window.
    /// </summary>
    [Description("The size of the sliding window.")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The stride of the sliding window.
    /// </summary>
    [Description("The stride of the sliding window.")]
    public long? Stride { get; set; } = null;

    /// <summary>
    /// The implicit negative infinity padding to be added on both sides.
    /// </summary>
    [Description("The implicit negative infinity padding to be added on both sides.")]
    public long? Padding { get; set; } = null;

    /// <summary>
    /// The spacing between kernel elements.
    /// </summary>
    [Description("The spacing between kernel elements.")]
    public long? Dilation { get; set; } = null;

    /// <summary>
    /// If set to true, will use ceil instead of floor to compute the output shape.
    /// </summary>
    [Description("If set to true, will use ceil instead of floor to compute the output shape.")]
    public bool CeilMode { get; set; } = false;

    /// <summary>
    /// Creates a MaxPool1d module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(MaxPool1d(KernelSize, Stride, Padding, Dilation, CeilMode));
    }

    /// <summary>
    /// Creates a MaxPool1d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => MaxPool1d(KernelSize, Stride, Padding, Dilation, CeilMode));
    }
}
