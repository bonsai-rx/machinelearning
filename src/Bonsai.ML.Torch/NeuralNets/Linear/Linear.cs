using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Linear;

/// <summary>
/// Represents an operator that creates a Linear module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Linear.html"/> for more information.
/// </remarks>
[Description("Creates a Linear module.")]
public class Linear
{
    /// <summary>
    /// The size of each input sample.
    /// </summary>
    [Description("The size of each input sample.")]
    public long InputSize { get; set; }

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
    /// Creates a Linear module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Linear> Process()
    {
        return Observable.Return(nn.Linear(InputSize, OutputSize, HasBias, Device, Type));
    }

    /// <summary>
    /// Creates a Linear module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Linear> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => nn.Linear(InputSize, OutputSize, HasBias, Device, Type));
    }
}
