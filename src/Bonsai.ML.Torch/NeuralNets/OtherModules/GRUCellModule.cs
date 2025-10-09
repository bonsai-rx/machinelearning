using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.OtherModules;

/// <summary>
/// Creates a GRUCell module.
/// </summary>
[Combinator]
[Description("Creates a GRUCell module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class GRUCellModule
{
    /// <summary>
    /// The inputsize parameter for the GRUCell module.
    /// </summary>
    [Description("The inputsize parameter for the GRUCell module")]
    public long InputSize { get; set; }

    /// <summary>
    /// The hiddensize parameter for the GRUCell module.
    /// </summary>
    [Description("The hiddensize parameter for the GRUCell module")]
    public long HiddenSize { get; set; }

    /// <summary>
    /// If true, adds a learnable bias to the output.
    /// </summary>
    [Description("If true, adds a learnable bias to the output")]
    public bool Bias { get; set; } = true;

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
    /// Generates an observable sequence that creates a GRUCellModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(GRUCell(InputSize, HiddenSize, Bias, Device, Type));
    }
}
