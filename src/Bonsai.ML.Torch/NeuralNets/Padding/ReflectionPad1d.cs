using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Padding;

/// <summary>
/// Represents an operator that creates a 1D reflection padding module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad1d.html"/> for more information.
/// </remarks>
[Description("Creates a 1D reflection padding module.")]
public class ReflectionPad1d
{
    /// <summary>
    /// The size of the padding.
    /// </summary>
    [Description("The size of the padding.")]
    [TypeConverter(typeof(ValueTupleConverter<long, long>))]
    public (long, long) PaddingSize { get; set; }

    /// <summary>
    /// Creates a 1D reflection padding module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(ReflectionPad1d(PaddingSize));
    }

    /// <summary>
    /// Creates a 1D reflection padding module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => ReflectionPad1d(PaddingSize));
    }
}
