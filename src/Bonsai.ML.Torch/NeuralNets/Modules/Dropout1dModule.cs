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
/// Creates a Dropout1d module module.
/// </summary>
[Combinator]
[Description("Creates a Dropout1d module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Dropout1dModule
{
    /// <summary>
    /// The p parameter for the Dropout1d module.
    /// </summary>
    [Description("The p parameter for the Dropout1d module")]
    public double P { get; set; } = 0.5;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a Dropout1d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Dropout1d(P, Inplace));
    }
}
