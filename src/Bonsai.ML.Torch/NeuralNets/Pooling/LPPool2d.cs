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
/// Represents an operator that creates a 2D power average pooling (LPPool2d) module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.LPPool2d.html"/> for more information.
/// </remarks>
[Description("Creates a 2D power average pooling (LPPool2d) module.")]
public class LPPool2d
{
    /// <summary>
    /// The degree of the norm.
    /// </summary>
    [Description("The degree of the norm.")]
    public double Norm { get; set; }

    /// <summary>
    /// The size of the window.
    /// </summary>
    [Description("The size of the window.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] KernelSize { get; set; }

    /// <summary>
    /// The stride of the window.
    /// </summary>
    [Description("The stride of the window.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] Stride { get; set; } = null;

    /// <summary>
    /// If set to true, will use ceil instead of floor to compute the output shape.
    /// </summary>
    [Description("If set to true, will use ceil instead of floor to compute the output shape.")]
    public bool CeilMode { get; set; } = false;

    /// <summary>
    /// Creates a LPPool2d module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.LPPool2d> Process()
    {
        return Observable.Return(LPPool2d(Norm, KernelSize, Stride, CeilMode));
    }

    /// <summary>
    /// Creates a LPPool2d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.LPPool2d> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => LPPool2d(Norm, KernelSize, Stride, CeilMode));
    }
}
