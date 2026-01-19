using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Padding;

/// <summary>
/// Represents an operator that creates a 3D reflection padding module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad3d.html"/> for more information.
/// </remarks>
[Description("Creates a 3D reflection padding module.")]
public class ReflectionPad3d
{
    /// <summary>
    /// The size of the padding.
    /// </summary>
    [Description("The size of the padding.")]
    [TypeConverter(typeof(ValueTupleConverter<long, long, long, long, long, long>))]
    public (long, long, long, long, long, long) PaddingSize { get; set; }

    /// <summary>
    /// Creates a 3D reflection padding module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.ReflectionPad3d> Process()
    {
        return Observable.Return(ReflectionPad3d(PaddingSize));
    }

    /// <summary>
    /// Creates a 3D reflection padding module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.ReflectionPad3d> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => ReflectionPad3d(PaddingSize));
    }
}
