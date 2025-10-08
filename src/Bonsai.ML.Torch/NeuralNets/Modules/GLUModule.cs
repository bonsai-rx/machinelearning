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
/// Creates a GLU module module.
/// </summary>
[Combinator]
[Description("Creates a GLU module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class GLUModule
{
    /// <summary>
    /// The dim parameter for the GLU module.
    /// </summary>
    [Description("The dim parameter for the GLU module")]
    public long Dim { get; set; } = -1;

    /// <summary>
    /// Generates an observable sequence that creates a GLU module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(GLU(Dim));
    }
}
