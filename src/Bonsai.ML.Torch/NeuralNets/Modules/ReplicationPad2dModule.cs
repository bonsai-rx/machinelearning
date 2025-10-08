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
/// Creates a ReplicationPad2d module module.
/// </summary>
[Combinator]
[Description("Creates a ReplicationPad2d module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ReplicationPad2dModule
{
    /// <summary>
    /// The padding parameter for the ReplicationPad2d module.
    /// </summary>
    [Description("The padding parameter for the ReplicationPad2d module")]
    public long Padding { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a ReplicationPad2d module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(ReplicationPad2d(Padding));
    }
}
