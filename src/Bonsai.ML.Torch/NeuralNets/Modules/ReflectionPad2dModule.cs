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
/// Creates a ReflectionPad2d module.
/// </summary>
[Combinator]
[Description("Creates a ReflectionPad2d module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ReflectionPad2dModule
{
    /// <summary>
    /// The padding parameter for the ReflectionPad2d module.
    /// </summary>
    [Description("The padding parameter for the ReflectionPad2d module")]
    public long Padding { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a ReflectionPad2dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(ReflectionPad2d(Padding));
    }
}
