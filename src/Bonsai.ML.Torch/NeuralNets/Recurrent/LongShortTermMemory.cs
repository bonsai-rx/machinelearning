using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Recurrent;

/// <summary>
/// Represents an operator that creates a multi-layer long short-term memory (LSTM) module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html"/> for more information.
/// </remarks>
[Description("Creates a multi-layer long short-term memory (LSTM) module.")]
public class LongShortTermMemory
{
    /// <summary>
    /// The number of expected features in the input.
    /// </summary>
    [Description("The number of expected features in the input.")]
    public long InputSize { get; set; }

    /// <summary>
    /// The number of features in the hidden state.
    /// </summary>
    [Description("The number of features in the hidden state.")]
    public long HiddenSize { get; set; }

    /// <summary>
    /// The number of recurrent layers.
    /// </summary>
    [Description("The number of recurrent layers.")]
    public long NumLayers { get; set; } = 1;

    /// <summary>
    /// If set to false, then the layer will not use bias weights.
    /// </summary>
    [Description("If set to false, then the layer will not use bias weights.")]
    public bool Bias { get; set; } = true;

    /// <summary>
    /// If set to true, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
    /// </summary>
    [Description("If set to true, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).")]
    public bool BatchFirst { get; set; } = false;

    /// <summary>
    /// If non-zero, this introduces a Dropout layer with the provided dropout probability on the outputs of each LSTM layer except the last layer.
    /// </summary>
    [Description("If non-zero, this introduces a Dropout layer with the provided dropout probability on the outputs of each LSTM layer except the last layer.")]
    public double Dropout { get; set; } = 0D;

    /// <summary>
    /// If set to true, becomes a bidirectional LSTM.
    /// </summary>
    [Description("If set to true, becomes a bidirectional LSTM.")]
    public bool Bidirectional { get; set; } = false;

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
    /// Creates an LSTM module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, (Tensor, Tensor)?, (Tensor, Tensor, Tensor)>> Process()
    {
        return Observable.Return(LSTM(InputSize, HiddenSize, NumLayers, Bias, BatchFirst, Dropout, Bidirectional, Device, Type));
    }

    /// <summary>
    /// Creates an LSTM module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, (Tensor, Tensor)?, (Tensor, Tensor, Tensor)>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => LSTM(InputSize, HiddenSize, NumLayers, Bias, BatchFirst, Dropout, Bidirectional, Device, Type));
    }
}
