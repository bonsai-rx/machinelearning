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
/// Creates a Linear transformation layer module.
/// </summary>
[Combinator]
[Description("Creates a Linear transformation layer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LinearModule
{
    /// <summary>
    /// The inputsize parameter for the Linear module.
    /// </summary>
    [Description("The inputsize parameter for the Linear module")]
    public long InputSize { get; set; }

    /// <summary>
    /// The outputsize parameter for the Linear module.
    /// </summary>
    [Description("The outputsize parameter for the Linear module")]
    public long OutputSize { get; set; }

    /// <summary>
    /// The hasbias parameter for the Linear module.
    /// </summary>
    [Description("The hasbias parameter for the Linear module")]
    public bool HasBias { get; set; } = true;

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
    /// Generates an observable sequence that creates a Linear module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Linear(InputSize, OutputSize, HasBias, Device, Type));
    }
}
