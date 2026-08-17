using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Vision;

/// <summary>
/// Represents an operator that creates a pixel unshuffle module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html"/> for more information.
/// </remarks>
[Description("Creates a pixel unshuffle module.")]
public class PixelUnshuffle
{
    /// <summary>
    /// The factor to decrease spatial resolution by.
    /// </summary>
    [Description("The factor to decrease spatial resolution by.")]
    public long DownscaleFactor { get; set; }

    /// <summary>
    /// Creates a pixel unshuffle module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.PixelUnshuffle> Process()
    {
        return Observable.Return(PixelUnshuffle(DownscaleFactor));
    }

    /// <summary>
    /// Creates a pixel unshuffle module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.PixelUnshuffle> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => PixelUnshuffle(DownscaleFactor));
    }
}
