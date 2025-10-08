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
/// Creates a Unflatten module module.
/// </summary>
[Combinator]
[Description("Creates a Unflatten module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class UnflattenModule
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
    /// Generates an observable sequence that creates a Unflatten module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Unflatten(Dim, UnflattenedSize));
    }
}
