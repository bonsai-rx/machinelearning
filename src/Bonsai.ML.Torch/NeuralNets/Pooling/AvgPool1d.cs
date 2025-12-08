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
/// Represents an operator that creates a 1D average pooling module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html"/> for more information.
/// </remarks>
[Description("Creates a 1D average pooling module.")]
public class AvgPool1d
{
    /// <summary>
    /// The size of the window.
    /// </summary>
    [Description("The size of the window.")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The stride of the window.
    /// </summary>
    [Description("The stride of the window.")]
    public long? Stride { get; set; } = null;

    /// <summary>
    /// The implicit zero padding to be added on both sides.
    /// </summary>
    [Description("The implicit zero padding to be added on both sides.")]
    public long Padding { get; set; } = 0;

    /// <summary>
    /// If set to true, will use ceil instead of floor to compute the output shape.
    /// </summary>
    [Description("If set to true, will use ceil instead of floor to compute the output shape.")]
    public bool CeilMode { get; set; } = false;

    /// <summary>
    /// If set to true, will include the zero-padding in the averaging calculation.
    /// </summary>
    [Description("If set to true, will include the zero-padding in the averaging calculation.")]
    public bool CountIncludePad { get; set; } = true;

    /// <summary>
    /// Creates an AvgPool1d module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(AvgPool1d(KernelSize, Stride, Padding, CeilMode, CountIncludePad));
    }

    /// <summary>
    /// Creates an AvgPool1d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => AvgPool1d(KernelSize, Stride, Padding, CeilMode, CountIncludePad));
    }
}
