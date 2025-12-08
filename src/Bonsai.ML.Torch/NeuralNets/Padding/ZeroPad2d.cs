using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Padding;

/// <summary>
/// Represents an operator that creates a 2D zero padding module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad2d.html"/> for more information.
/// </remarks>
[Description("Creates a 2D zero padding module.")]
public class ZeroPad2d
{
    /// <summary>
    /// The size of the padding.
    /// </summary>
    [Description("The size of the padding.")]
    [TypeConverter(typeof(ValueTupleConverter<long, long, long, long>))]
    public (long, long, long, long) PaddingSize { get; set; }

    /// <summary>
    /// Creates a 2D zero padding module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(ZeroPad2d(PaddingSize));
    }

    /// <summary>
    /// Creates a 2D zero padding module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => ZeroPad2d(PaddingSize));
    }
}
