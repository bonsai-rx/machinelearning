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
/// Creates a ReplicationPad1d module module.
/// </summary>
[Combinator]
[Description("Creates a ReplicationPad1d module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ReplicationPad1dModule
{
    /// <summary>
    /// The padding parameter for the ReplicationPad1d module.
    /// </summary>
    [Description("The padding parameter for the ReplicationPad1d module")]
    public long Padding { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a ReplicationPad1d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(ReplicationPad1d(Padding));
    }
}
