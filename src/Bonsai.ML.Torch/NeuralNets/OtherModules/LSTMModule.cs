using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.OtherModules;

/// <summary>
/// Creates a Long Short-Term Memory layer module.
/// </summary>
[Combinator]
[Description("Creates a Long Short-Term Memory layer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LSTMModule
{
    /// <summary>
    /// The inputsize parameter for the LSTM module.
    /// </summary>
    [Description("The inputsize parameter for the LSTM module")]
    public long InputSize { get; set; }

    /// <summary>
    /// The hiddensize parameter for the LSTM module.
    /// </summary>
    [Description("The hiddensize parameter for the LSTM module")]
    public long HiddenSize { get; set; }

    /// <summary>
    /// The numlayers parameter for the LSTM module.
    /// </summary>
    [Description("The numlayers parameter for the LSTM module")]
    public long NumLayers { get; set; } = 1;

    /// <summary>
    /// If true, adds a learnable bias to the output.
    /// </summary>
    [Description("If true, adds a learnable bias to the output")]
    public bool Bias { get; set; } = true;

    /// <summary>
    /// The batchfirst parameter for the LSTM module.
    /// </summary>
    [Description("The batchfirst parameter for the LSTM module")]
    public bool BatchFirst { get; set; } = false;

    /// <summary>
    /// The dropout parameter for the LSTM module.
    /// </summary>
    [Description("The dropout parameter for the LSTM module")]
    public double Dropout { get; set; } = 0;

    /// <summary>
    /// The bidirectional parameter for the LSTM module.
    /// </summary>
    [Description("The bidirectional parameter for the LSTM module")]
    public bool Bidirectional { get; set; } = false;

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
    /// Generates an observable sequence that creates a LSTM module.
    /// </summary>
    public IObservable<IModule<Tensor, (Tensor, Tensor)?, (Tensor, Tensor, Tensor)>> Process()
    {
        return Observable.Return(LSTM(InputSize, HiddenSize, NumLayers, Bias, BatchFirst, Dropout, Bidirectional, Device, Type));
    }
}
