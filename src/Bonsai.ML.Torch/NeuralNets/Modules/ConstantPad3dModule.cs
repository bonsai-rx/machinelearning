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
/// Creates a ConstantPad3d module module.
/// </summary>
[Combinator]
[Description("Creates a ConstantPad3d module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ConstantPad3dModule
{
    /// <summary>
    /// The padding parameter for the ConstantPad3d module.
    /// </summary>
    [Description("The padding parameter for the ConstantPad3d module")]
    public long Padding { get; set; }

    /// <summary>
    /// The value parameter for the ConstantPad3d module.
    /// </summary>
    [Description("The value parameter for the ConstantPad3d module")]
    public double Value { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a ConstantPad3d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(ConstantPad3d(Padding, Value));
    }
}
