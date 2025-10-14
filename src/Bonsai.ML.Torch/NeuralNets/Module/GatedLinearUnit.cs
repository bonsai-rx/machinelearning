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
/// Creates a GLU module.
/// </summary>
[Combinator]
[Description("Creates a GLU module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class GatedLinearUnit
{
    /// <summary>
    /// The dim parameter for the GLU module.
    /// </summary>
    [Description("The dim parameter for the GLU module")]
    public long Dim { get; set; } = -1;

    /// <summary>
    /// Generates an observable sequence that creates a GLUModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(GLU(Dim));
    }
}
