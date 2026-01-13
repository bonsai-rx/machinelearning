using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Shuffle;

/// <summary>
/// Represents an operator that creates a channel shuffle module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.ChannelShuffle.html"/> for more information.
/// </remarks>
[Description("Creates a channel shuffle module.")]
public class ChannelShuffle
{
    /// <summary>
    /// The number of groups to divide the channels into.
    /// </summary>
    [Description("The number of groups to divide the channels into.")]
    public long Groups { get; set; }

    /// <summary>
    /// Creates a channel shuffle module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(ChannelShuffle(Groups));
    }

    /// <summary>
    /// Creates a channel shuffle module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => ChannelShuffle(Groups));
    }
}
