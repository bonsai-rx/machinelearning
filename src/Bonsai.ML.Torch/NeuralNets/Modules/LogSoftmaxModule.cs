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
/// Creates a LogSoftmax activation function module.
/// </summary>
[Combinator]
[Description("Creates a LogSoftmax activation function module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LogSoftmaxModule
{
    /// <summary>
    /// The dim parameter for the LogSoftmax module.
    /// </summary>
    [Description("The dim parameter for the LogSoftmax module")]
    public long Dim { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a LogSoftmax module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(LogSoftmax(Dim));
    }
}
