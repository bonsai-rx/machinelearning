using Bonsai;
using static TorchSharp.torch;
using TorchSharp;
using System;
using System.Reactive.Linq;
using System.Xml.Serialization;
using System.ComponentModel;

namespace Bonsai.ML.Torch.Random;

/// <summary>
/// Sets the global random seed for TorchSharp and creates a random number generator with the specified seed.
/// </summary>
[Combinator]
[Description("Sets the global random seed for TorchSharp and creates a random number generator with the specified seed.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ManualSeed
{
    /// <summary>
    /// The seed for the random number generator.
    /// </summary>
    public long Seed { get; set; } = 0;

    /// <summary>
    /// Creates a random number generator with the specified seed and device.
    /// </summary>
    /// <returns></returns>
    public IObservable<torch.Generator> Process()
    {
        return Observable.Return(manual_seed(Seed));
    }
}
