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
/// Creates a Softmax activation function.
/// </summary>
[Combinator]
[Description("Creates a Softmax activation function.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SoftMax
{
    /// <summary>
    /// The dim parameter for the Softmax module.
    /// </summary>
    [Description("The dim parameter for the Softmax module")]
    public long Dim { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a SoftmaxModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(Softmax(Dim));
    }
}
