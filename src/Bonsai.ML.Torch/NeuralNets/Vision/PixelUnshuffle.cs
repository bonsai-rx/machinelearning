using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Vision;

/// <summary>
/// Creates a PixelUnshuffle module.
/// </summary>
[Combinator]
[Description("Creates a PixelUnshuffle module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PixelUnshuffle
{
    /// <summary>
    /// The downscalefactor parameter for the PixelUnshuffle module.
    /// </summary>
    [Description("The downscalefactor parameter for the PixelUnshuffle module")]
    public long DownscaleFactor { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a PixelUnshuffleModule module.
    /// </summary>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(PixelUnshuffle(DownscaleFactor));
    }
}
