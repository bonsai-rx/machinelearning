using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Modules;

/// <summary>
/// Creates a Upsample module.
/// </summary>
[Combinator]
[Description("Creates a Upsample module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class UpsampleModule
{
    /// <summary>
    /// The size parameter for the Upsample module.
    /// </summary>
    [Description("The size parameter for the Upsample module")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] Size { get; set; } = null;

    /// <summary>
    /// The scale_factor parameter for the Upsample module.
    /// </summary>
    [Description("The scale_factor parameter for the Upsample module")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public double[] ScaleFactor { get; set; } = null;

    /// <summary>
    /// The mode parameter for the Upsample module.
    /// </summary>
    [Description("The mode parameter for the Upsample module")]
    public UpsampleMode Mode { get; set; } = UpsampleMode.Nearest;

    /// <summary>
    /// The align_corners parameter for the Upsample module.
    /// </summary>
    [Description("The align_corners parameter for the Upsample module")]
    public bool? AlignCorners { get; set; } = null;

    /// <summary>
    /// Generates an observable sequence that creates a UpsampleModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Upsample(Size, ScaleFactor, Mode, AlignCorners));
    }
}
