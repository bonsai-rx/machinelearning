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
/// Creates a Bilinear module.
/// </summary>
[Combinator]
[Description("Creates a Bilinear module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class BilinearModule
{
    /// <summary>
    /// The in1features parameter for the Bilinear module.
    /// </summary>
    [Description("The in1features parameter for the Bilinear module")]
    public long In1Features { get; set; }

    /// <summary>
    /// The in2features parameter for the Bilinear module.
    /// </summary>
    [Description("The in2features parameter for the Bilinear module")]
    public long In2Features { get; set; }

    /// <summary>
    /// The outputsize parameter for the Bilinear module.
    /// </summary>
    [Description("The outputsize parameter for the Bilinear module")]
    public long OutputSize { get; set; }

    /// <summary>
    /// The hasbias parameter for the Bilinear module.
    /// </summary>
    [Description("The hasbias parameter for the Bilinear module")]
    public bool HasBias { get; set; } = true;

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
    /// Generates an observable sequence that creates a BilinearModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(Bilinear(In1Features, In2Features, OutputSize, HasBias, Device, Type));
    }
}
