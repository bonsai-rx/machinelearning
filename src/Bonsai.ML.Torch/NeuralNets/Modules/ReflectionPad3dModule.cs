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
/// Creates a ReflectionPad3d module.
/// </summary>
[Combinator]
[Description("Creates a ReflectionPad3d module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ReflectionPad3dModule
{
    /// <summary>
    /// The padding parameter for the ReflectionPad3d module.
    /// </summary>
    [Description("The padding parameter for the ReflectionPad3d module")]
    public long Padding { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a ReflectionPad3dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(ReflectionPad3d(Padding));
    }
}
