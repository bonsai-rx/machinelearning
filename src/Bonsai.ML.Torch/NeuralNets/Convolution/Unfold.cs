using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Convolution;

/// <summary>
/// Represents an operator that creates an Unfold module.
/// </summary>
[Description("Creates an Unfold module.")]
public class Unfold
{
    /// <summary>
    /// The kernel_size parameter for the Unfold module.
    /// </summary>
    [Description("The kernel_size parameter for the Unfold module")]
    public long KernelSize { get; set; }

    /// <summary>
    /// The dilation parameter for the Unfold module.
    /// </summary>
    [Description("The dilation parameter for the Unfold module")]
    public long Dilation { get; set; } = 1;

    /// <summary>
    /// The padding parameter for the Unfold module.
    /// </summary>
    [Description("The padding parameter for the Unfold module")]
    public long Padding { get; set; } = 0;

    /// <summary>
    /// The stride parameter for the Unfold module.
    /// </summary>
    [Description("The stride parameter for the Unfold module")]
    public long Stride { get; set; } = 1;

    /// <summary>
    /// Creates an Unfold module.
    /// </summary>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Unfold(KernelSize, Dilation, Padding, Stride));
    }

    /// <summary>
    /// Creates an Unfold module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Unfold(KernelSize, Dilation, Padding, Stride));
    }
}
