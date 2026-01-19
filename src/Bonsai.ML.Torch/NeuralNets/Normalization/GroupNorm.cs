using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Normalization;

/// <summary>
/// Represents an operator that creates a group normalization module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html"/> for more information.
/// </remarks>
[Description("Creates a group normalization module.")]
public class GroupNorm
{
    /// <summary>
    /// The number of groups to separate the channels into.
    /// </summary>
    [Description("The number of groups to separate the channels into.")]
    public long NumGroups { get; set; }

    /// <summary>
    /// The number of channels in the input.
    /// </summary>
    [Description("The number of channels in the input.")]
    public long NumChannels { get; set; }

    /// <summary>
    /// The value added to the denominator for numerical stability.
    /// </summary>
    [Description("The value added to the denominator for numerical stability.")]
    public double Eps { get; set; } = 1E-05D;

    /// <summary>
    /// A boolean value that when set to true, this module has learnable per-channel affine parameters initialized to ones (for weights) and zeros (for biases).
    /// </summary>
    [Description("A boolean value that when set to true, this module has learnable per-channel affine parameters initialized to ones (for weights) and zeros (for biases).")]
    public bool Affine { get; set; } = true;

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
    /// Creates a GroupNorm module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.GroupNorm> Process()
    {
        return Observable.Return(GroupNorm(NumGroups, NumChannels, Eps, Affine, Device, Type));
    }

    /// <summary>
    /// Creates a GroupNorm module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.GroupNorm> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => GroupNorm(NumGroups, NumChannels, Eps, Affine, Device, Type));
    }
}
