using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Linear;

/// <summary>
/// Represents an operator that creates a Bilinear module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Bilinear.html"/> for more information.
/// </remarks>
[Description("Creates a Bilinear module.")]
public class Bilinear
{
    /// <summary>
    /// The size of each first input sample.
    /// </summary>
    [Description("The size of each first input sample.")]
    public long In1Features { get; set; }

    /// <summary>
    /// The size of each second input sample.
    /// </summary>
    [Description("The size of each second input sample.")]
    public long In2Features { get; set; }

    /// <summary>
    /// The size of each output sample.
    /// </summary>
    [Description("The size of each output sample.")]
    public long OutputSize { get; set; }

    /// <summary>
    /// Determines whether the layer will learn an additive bias.
    /// </summary>
    [Description("Determines whether the layer will learn an additive bias.")]
    public bool HasBias { get; set; } = true;

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
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Creates a Bilinear module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Bilinear> Process()
    {
        return Observable.Return(Bilinear(In1Features, In2Features, OutputSize, HasBias, Device, Type));
    }

    /// <summary>
    /// Creates a Bilinear module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Bilinear> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Bilinear(In1Features, In2Features, OutputSize, HasBias, Device, Type));
    }
}
