using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Modules;

/// <summary>
/// Creates a PixelShuffle module.
/// </summary>
[Combinator]
[Description("Creates a PixelShuffle module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PixelShuffleModule
{
    /// <summary>
    /// The upscalefactor parameter for the PixelShuffle module.
    /// </summary>
    [Description("The upscalefactor parameter for the PixelShuffle module")]
    public long UpscaleFactor { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a PixelShuffleModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor>> Process()
    {
        return Observable.Return(PixelShuffle(UpscaleFactor));
    }
}
