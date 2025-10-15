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
/// Creates a 1D adaptive average pooling layer.
/// </summary>
[Combinator]
[Description("Creates a 1D adaptive average pooling layer.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class BatchNormalization
{
    /// <summary>
    /// The number of dimensions for the AdaptiveAvgPool module.
    /// </summary>
    [Description("The number of dimensions for the AdaptiveAvgPool module")]
    public Dimensions Dimensions { get; set; } = Dimensions.One;

    /// <summary>
    /// The features parameter for the BatchNorm1d module.
    /// </summary>
    [Description("The features parameter for the BatchNorm1d module")]
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
    public bool Affine { get; set; } = true;

    /// <summary>
    /// The track_running_stats parameter for the BatchNorm1d module.
    /// </summary>
    [Description("The track_running_stats parameter for the BatchNorm1d module")]
    public bool TrackRunningStats { get; set; } = true;

    /// <summary>
    /// The desired device of returned tensor.
    /// </summary>
    [XmlIgnore]
    [Description("The desired device of returned tensor")]
    public Device Device { get; set; } = null;

    /// <summary>
    /// The desired data type of returned tensor.
    /// </summary>
    [Description("The desired data type of returned tensor")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a AdaptiveAvgPool1dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Dimensions switch
        {
            Dimensions.One => Observable.Return(BatchNorm1d(Features, Eps, Momentum, Affine, TrackRunningStats, Device, Type)),
            Dimensions.Two => Observable.Return(BatchNorm2d(Features, Eps, Momentum, Affine, TrackRunningStats, Device, Type)),
            Dimensions.Three => Observable.Return(BatchNorm3d(Features, Eps, Momentum, Affine, TrackRunningStats, Device, Type)),
            _ => throw new InvalidOperationException("The specified number of dimensions is not supported."),
        };
    }
}
