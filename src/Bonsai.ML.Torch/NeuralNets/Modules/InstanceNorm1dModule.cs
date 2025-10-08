using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Modules;

/// <summary>
/// Creates a 1D instance normalization module.
/// </summary>
[Combinator]
[Description("Creates a 1D instance normalization module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class InstanceNorm1dModule
{
    /// <summary>
    /// The features parameter for the InstanceNorm1d module.
    /// </summary>
    [Description("The features parameter for the InstanceNorm1d module")]
    public long Features { get; set; }

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-05;

    /// <summary>
    /// The value used for the running_mean and running_var computation.
    /// </summary>
    [Description("The value used for the running_mean and running_var computation")]
    public double Momentum { get; set; } = 0.1;

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
    [XmlIgnore]
    public Device Device { get; set; } = null;

    /// <summary>
    /// The desired data type of returned tensor.
    /// </summary>
    [Description("The desired data type of returned tensor")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a InstanceNorm1d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(InstanceNorm1d(Features, Eps, Momentum, Affine, TrackRunningStats, Device, Type));
    }
}
