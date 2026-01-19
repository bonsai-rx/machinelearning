using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Pooling;

/// <summary>
/// Represents an operator that creates a 3D adaptive average pooling module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html"/> for more information.
/// </remarks>
[Description("Creates a 3D adaptive average pooling module.")]
public class AdaptiveAvgPool3d
{
    /// <summary>
    /// The output size.
    /// </summary>
    [Description("The output size.")]
    [TypeConverter(typeof(ValueTupleConverter<long, long, long>))]
    public (long, long, long) OutputSize { get; set; }

    /// <summary>
    /// Creates an AdaptiveAvgPool3d module.
    /// </summary>
    public IObservable<TorchSharp.Modules.AdaptiveAvgPool3d> Process()
    {
        return Observable.Return(AdaptiveAvgPool3d(OutputSize));
    }

    /// <summary>
    /// Creates an AdaptiveAvgPool3d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.AdaptiveAvgPool3d> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => AdaptiveAvgPool3d(OutputSize));
    }
}
