using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Module;

/// <summary>
/// Creates a Unflatten module.
/// </summary>
[Combinator]
[Description("Creates a Unflatten module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Unflatten
{
    /// <summary>
    /// The dim parameter for the Unflatten module.
    /// </summary>
    [Description("The dim parameter for the Unflatten module")]
    public long Dim { get; set; }

    /// <summary>
    /// The unflattenedsize parameter for the Unflatten module.
    /// </summary>
    [Description("The unflattenedsize parameter for the Unflatten module")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] UnflattenedSize { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a UnflattenModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Unflatten(Dim, UnflattenedSize));
    }
}
