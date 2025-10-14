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
/// Creates a ZeroPad2d module.
/// </summary>
[Combinator]
[Description("Creates a ZeroPad2d module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ZeroPad2dModule
{
    /// <summary>
    /// The padding parameter for the ZeroPad2d module.
    /// </summary>
    [Description("The padding parameter for the ZeroPad2d module")]
    public long Padding { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a ZeroPad2dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(ZeroPad2d(Padding));
    }
}
