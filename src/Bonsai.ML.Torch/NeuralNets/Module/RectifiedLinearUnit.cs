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
/// Creates a ReLU activation function.
/// </summary>
[Combinator]
[Description("Creates a ReLU activation function.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class RectifiedLinearUnit
{
    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// If set to true, will use the bounded version of ReLU with a maximum value of 6.
    /// </summary>
    [Description("If set to true, will use the bounded version of ReLU with a maximum value of 6")]
    public bool Bounded { get; set; } = false;

    /// <summary>
    /// Generates an observable sequence that creates a RectifiedLinearUnit module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        if (Bounded)
            return Observable.Return(ReLU6(Inplace));
        else
            return Observable.Return(ReLU(Inplace));
    }
}
