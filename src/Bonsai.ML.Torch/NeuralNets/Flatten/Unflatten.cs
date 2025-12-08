using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Flatten;

/// <summary>
/// Creates an Unflatten module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.modules.flatten.Unflatten.html"/> for more information.
/// </remarks>
[Description("Creates an Unflatten module.")]
public class Unflatten
{
    /// <summary>
    /// The dimension to unflatten.
    /// </summary>
    [Description("The dimension to unflatten.")]
    public long Dim { get; set; }

    /// <summary>
    /// The new shape of the unflattened dimension.
    /// </summary>
    [Description("The new shape of the unflattened dimension.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] UnflattenedSize { get; set; }

    /// <summary>
    /// Creates an Unflatten module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Unflatten(Dim, UnflattenedSize));
    }

    /// <summary>
    /// Creates an Unflatten module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Unflatten(Dim, UnflattenedSize));
    }
}
