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
/// Creates a 3D dropout regularization layer module.
/// </summary>
[Combinator]
[Description("Creates a 3D dropout regularization layer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Dropout3dModule
{
    /// <summary>
    /// The p parameter for the Dropout3d module.
    /// </summary>
    [Description("The p parameter for the Dropout3d module")]
    public double P { get; set; } = 0.5;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a Dropout3d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Dropout3d(P, Inplace));
    }
}
