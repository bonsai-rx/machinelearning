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
/// Creates a PixelUnshuffle module module.
/// </summary>
[Combinator]
[Description("Creates a PixelUnshuffle module module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PixelUnshuffleModule
{
    /// <summary>
    /// The downscalefactor parameter for the PixelUnshuffle module.
    /// </summary>
    [Description("The downscalefactor parameter for the PixelUnshuffle module")]
    public long DownscaleFactor { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a PixelUnshuffle module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(PixelUnshuffle(DownscaleFactor));
    }
}
