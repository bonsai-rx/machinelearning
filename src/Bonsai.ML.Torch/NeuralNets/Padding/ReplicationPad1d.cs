using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Padding;

/// <summary>
/// Represents an operator that creates a 1D replication padding module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad1d.html"/> for more information.
/// </remarks>
[Description("Creates a 1D replication padding module.")]
public class ReplicationPad1d
{
    /// <summary>
    /// The size of the padding.
    /// </summary>
    [Description("The size of the padding.")]
    [TypeConverter(typeof(ValueTupleConverter<long, long>))]
    public (long, long) PaddingSize { get; set; }

    /// <summary>
    /// Creates a 1D replication padding module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.ReplicationPad1d> Process()
    {
        return Observable.Return(ReplicationPad1d(PaddingSize));
    }

    /// <summary>
    /// Creates a 1D replication padding module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.ReplicationPad1d> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => ReplicationPad1d(PaddingSize));
    }
}
