using Bonsai;
using static TorchSharp.torch;
using TorchSharp;
using System;
using System.Reactive.Linq;
using System.Xml.Serialization;
using System.ComponentModel;

namespace Bonsai.ML.Torch.Random;

/// <summary>
/// Creates a random number generator with the specified seed and device.
/// </summary>
[Combinator]
[Description("Creates a random number generator with the specified seed and device.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Generator
{
    /// <summary>
    /// The device on which to create the generator.
    /// </summary>
    [XmlIgnore]
    public Device Device { get; set; }

    /// <summary>
    /// The seed for the random number generator.
    /// </summary>
    public ulong Seed { get; set; } = 0;

    /// <summary>
    /// Creates a random number generator with the specified seed and device.
    /// </summary>
    /// <returns></returns>
    public IObservable<torch.Generator> Process()
    {
        return Observable.Return(new torch.Generator(Seed, Device));
    }
}
