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
/// Creates a Group normalization.
/// </summary>
[Combinator]
[Description("Creates a Group normalization.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class GroupNormalization
{
    /// <summary>
    /// The num_groups parameter for the GroupNorm module.
    /// </summary>
    [Description("The num_groups parameter for the GroupNorm module")]
    public long NumGroups { get; set; }

    /// <summary>
    /// The num_channels parameter for the GroupNorm module.
    /// </summary>
    [Description("The num_channels parameter for the GroupNorm module")]
    public long NumChannels { get; set; }

    /// <summary>
    /// A value added to the denominator for numerical stability.
    /// </summary>
    [Description("A value added to the denominator for numerical stability")]
    public double Eps { get; set; } = 1E-05D;

    /// <summary>
    /// A boolean value that when set to true, this module has learnable affine parameters.
    /// </summary>
    [Description("A boolean value that when set to true, this module has learnable affine parameters")]
    public bool Affine { get; set; } = true;

    /// <summary>
    /// The desired device of returned tensor.
    /// </summary>
    [Description("The desired device of returned tensor")]
    public Device Device { get; set; } = null;

    /// <summary>
    /// The desired data type of returned tensor.
    /// </summary>
    [Description("The desired data type of returned tensor")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a GroupNormModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(GroupNorm(NumGroups, NumChannels, Eps, Affine, Device, Type));
    }
}
