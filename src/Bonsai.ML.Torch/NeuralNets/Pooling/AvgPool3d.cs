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
/// Represents an operator that creates a 3D average pooling layer.
/// </summary>
[Description("Creates a 3D average pooling layer.")]
public class AvgPool3d
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
    /// The ceiling mode.
    /// </summary>
    [Description("The ceiling mode.")]
    public bool CeilMode { get; set; } = false;

    /// <summary>
    /// The count include pad parameter.
    /// </summary>
    [Description("The count include pad parameter.")]
    public bool CountIncludePad { get; set; } = true;

    /// <summary>
    /// The divisor override.
    /// </summary>
    [Description("The divisor override.")]
    public long? DivisorOverride { get; set; } = null;

    /// <summary>
    /// Creates an AvgPool3d module.
    /// </summary>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(AvgPool3d(KernelSize, Stride, Padding, CeilMode, CountIncludePad, DivisorOverride));
    }

    /// <summary>
    /// Creates an AvgPool3d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => AvgPool3d(KernelSize, Stride, Padding, CeilMode, CountIncludePad, DivisorOverride));
    }
}
