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
/// Creates a Softmin module module.
/// </summary>
[Combinator]
[Description("Creates a Softmin module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SoftminModule
{
    /// <summary>
    /// The dim parameter for the Softmin module.
    /// </summary>
    [Description("The dim parameter for the Softmin module")]
    public long Dim { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a Softmin module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Softmin(Dim));
    }
}
