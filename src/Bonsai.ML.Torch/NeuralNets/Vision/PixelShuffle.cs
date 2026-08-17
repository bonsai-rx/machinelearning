using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Vision;

/// <summary>
/// Represents an operator that creates a pixel shuffle module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html"/> for more information.
/// </remarks>
[Description("Creates a pixel shuffle module.")]
public class PixelShuffle
{
    /// <summary>
    /// The factor to increase spatial resolution by.
    /// </summary>
    [Description("The factor to increase spatial resolution by.")]
    public long UpscaleFactor { get; set; }

    /// <summary>
    /// Creates a pixel shuffle module.
    /// </summary>
    public IObservable<TorchSharp.Modules.PixelShuffle> Process()
    {
        return Observable.Return(PixelShuffle(UpscaleFactor));
    }

    /// <summary>
    /// Creates a pixel shuffle module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.PixelShuffle> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => PixelShuffle(UpscaleFactor));
    }
}
