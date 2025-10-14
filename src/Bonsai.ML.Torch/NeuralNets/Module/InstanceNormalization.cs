using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Module;

/// <summary>
/// Creates a dropout layer.
/// </summary>
[Combinator]
[Description("Creates a fractional max pooling layer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class InstanceNormalization
{
    /// <summary>
    /// The number of dimensions for the AdaptiveAvgPool module.
    /// </summary>
    [Description("The number of dimensions for the AdaptiveAvgPool module")]
    public Dimensions Dimensions { get; set; } = Dimensions.Two;

    /// <summary>
    /// The features parameter for the InstanceNorm1d module.
    /// </summary>
    [Description("The features parameter for the InstanceNorm1d module")]
    public long Features { get; set; }

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-05D;

    /// <summary>
    /// The value used for the running_mean and running_var computation.
    /// </summary>
    [Description("The value used for the running_mean and running_var computation")]
    public double Momentum { get; set; } = 0.1D;

    /// <summary>
    /// A boolean value that when set to true, this module has learnable affine parameters.
    /// </summary>
    [Description("A boolean value that when set to true, this module has learnable affine parameters")]
    public bool Affine { get; set; } = false;

    /// <summary>
    /// The track_running_stats parameter for the InstanceNorm1d module.
    /// </summary>
    [Description("The track_running_stats parameter for the InstanceNorm1d module")]
    public bool TrackRunningStats { get; set; } = false;

    /// <summary>
    /// The desired device of returned tensor.
    /// </summary>
    [Description("The desired device of returned tensor")]
    public torch.Device Device { get; set; } = null;

    /// <summary>
    /// The desired data type of returned tensor.
    /// </summary>
    [Description("The desired data type of returned tensor")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a Dropout module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Dimensions switch
        {
            Dimensions.One => Observable.Return(InstanceNorm1d(Features, Eps, Momentum, Affine, TrackRunningStats, Device, Type)),
            Dimensions.Two => Observable.Return(InstanceNorm2d(Features, Eps, Momentum, Affine, TrackRunningStats, Device, Type)),
            Dimensions.Three => Observable.Return(InstanceNorm3d(Features, Eps, Momentum, Affine, TrackRunningStats, Device, Type)),
            _ => throw new InvalidOperationException("The specified number of dimensions is not supported."),
        };
    }
}
