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
/// Creates a ConstantPad2d module.
/// </summary>
[Combinator]
[Description("Creates a ConstantPad2d module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ConstantPad2dModule
{
    /// <summary>
    /// The padding parameter for the ConstantPad2d module.
    /// </summary>
    [Description("The padding parameter for the ConstantPad2d module")]
    public long Padding { get; set; }

    /// <summary>
    /// The value parameter for the ConstantPad2d module.
    /// </summary>
    [Description("The value parameter for the ConstantPad2d module")]
    public double Value { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a ConstantPad2dModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(ConstantPad2d(Padding, Value));
    }
}
