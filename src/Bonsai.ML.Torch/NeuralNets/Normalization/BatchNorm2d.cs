using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Normalization;

/// <summary>
/// Represents an operator that creates a 2D batch normalization module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html"/> for more information.
/// </remarks>
[Description("Creates a 2D batch normalization module.")]
public class BatchNorm2d
{
    /// <summary>
    /// The number of features of the input.
    /// </summary>
    [Description("The number of features of the input.")]
    public long Features { get; set; }

    /// <summary>
    /// The value added to the denominator for numerical stability.
    /// </summary>
    [Description("The value added to the denominator for numerical stability.")]
    public double Eps { get; set; } = 1E-05D;

    /// <summary>
    /// The value used for computing the running mean and running variance.
    /// </summary>
    [Description("The value used for computing the running mean and running variance.")]
    public double Momentum { get; set; } = 0.1D;

    /// <summary>
    /// If set to true, this module has learnable affine parameters.
    /// </summary>
    [Description("If set to true, this module has learnable affine parameters")]
    public bool Affine { get; set; } = true;

    /// <summary>
    /// If set to true, this module tracks the running mean and variance, otherwise, the module will use batch statistics.
    /// </summary>
    [Description("If set to true, this module tracks the running mean and variance, otherwise, the module will use batch statistics.")]
    public bool TrackRunningStats { get; set; } = true;

    /// <summary>
    /// The desired device of the returned tensor.
    /// </summary>
    [XmlIgnore]
    [Description("The desired device of the returned tensor.")]
    public Device Device { get; set; } = null;

    /// <summary>
    /// The desired data type of the returned tensor.
    /// </summary>
    [Description("The desired data type of the returned tensor.")]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Creates a BatchNorm2d module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.BatchNorm2d> Process()
    {
        return Observable.Return(BatchNorm2d(Features, Eps, Momentum, Affine, TrackRunningStats, Device, Type));
    }

    /// <summary>
    /// Creates a BatchNorm2d module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.BatchNorm2d> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => BatchNorm2d(Features, Eps, Momentum, Affine, TrackRunningStats, Device, Type));
    }
}
