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
/// Creates a ChannelShuffle module module.
/// </summary>
[Combinator]
[Description("Creates a ChannelShuffle module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ChannelShuffleModule
{
    /// <summary>
    /// The groups parameter for the ChannelShuffle module.
    /// </summary>
    [Description("The groups parameter for the ChannelShuffle module")]
    public long Groups { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a ChannelShuffle module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(ChannelShuffle(Groups));
    }
}
