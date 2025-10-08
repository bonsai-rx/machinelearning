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
/// Creates a PReLU module module.
/// </summary>
[Combinator]
[Description("Creates a PReLU module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PReLUModule
{
    /// <summary>
    /// The num_parameters parameter for the PReLU module.
    /// </summary>
    [Description("The num_parameters parameter for the PReLU module")]
    public long NumParameters { get; set; }

    /// <summary>
    /// The init parameter for the PReLU module.
    /// </summary>
    [Description("The init parameter for the PReLU module")]
    public double Init { get; set; } = 0.25;

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
    /// Generates an observable sequence that creates a PReLU module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(PReLU(NumParameters, Init, Device, Type));
    }
}
