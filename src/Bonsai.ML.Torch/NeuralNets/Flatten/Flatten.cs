using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Flatten;

/// <summary>
/// Represents an operator that creates a Flatten module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.modules.flatten.Flatten.html"/> for more information.
/// </remarks>
[Description("Creates a Flatten module.")]
public class Flatten
{
    /// <summary>
    /// The first dimension to flatten.
    /// </summary>
    [Description("The first dimension to flatten.")]
    public long StartDim { get; set; } = 1;

    /// <summary>
    /// The last dimension to flatten.
    /// </summary>
    [Description("The last dimension to flatten.")]
    public long EndDim { get; set; } = -1;

    /// <summary>
    /// Creates a Flatten module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(nn.Flatten(StartDim, EndDim));
    }

    /// <summary>
    /// Creates a Flatten module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => nn.Flatten(StartDim, EndDim));
    }
}
