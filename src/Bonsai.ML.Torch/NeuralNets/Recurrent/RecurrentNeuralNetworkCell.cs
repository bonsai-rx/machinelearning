using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Recurrent;

/// <summary>
/// Represents an operator that creates a recurrent neural network (RNN) cell.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html"/> for more information.
/// </remarks>
[Description("Creates a recurrent neural network (RNN) cell.")]
public class RecurrentNeuralNetworkCell
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
    /// The type of non-linearity to use.
    /// </summary>
    [Description("The type of non-linearity to use.")]
    public NonLinearities NonLinearity { get; set; } = NonLinearities.Tanh;

    /// <summary>
    /// If set to false, then the layer will not use bias weights.
    /// </summary>
    [Description("If set to false, then the layer will not use bias weights.")]
    public bool Bias { get; set; } = true;

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
    /// Creates an RNNCell module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.RNNCell> Process()
    {
        return Observable.Return(RNNCell(InputSize, HiddenSize, NonLinearity, Bias, Device, Type));
    }

    /// <summary>
    /// Creates an RNNCell module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.RNNCell> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => RNNCell(InputSize, HiddenSize, NonLinearity, Bias, Device, Type));
    }
}
