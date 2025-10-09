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
/// Creates a SiLU module.
/// </summary>
[Combinator]
[Description("Creates a SiLU module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SiLUModule
{
    /// <summary>
    /// Generates an observable sequence that creates a SiLUModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(SiLU());
    }
}
