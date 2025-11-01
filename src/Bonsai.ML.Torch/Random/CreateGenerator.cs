using Bonsai;
using static TorchSharp.torch;
using TorchSharp;
using System;
using System.Reactive.Linq;
using System.Xml.Serialization;
using System.ComponentModel;

namespace Bonsai.ML.Torch.Random;

/// <summary>
/// Creates a specific instance of a random number generator with the specified seed and device.
/// </summary>
[Combinator]
[Description("Creates a specific instance of a random number generator with the specified seed and device.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CreateGenerator
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
    public IObservable<Generator> Process()
    {
        return Observable.Return(new Generator(Seed, Device));
    }

    /// <summary>
    /// Generates an observable sequence of random number generators for each element of the input sequence.
    /// </summary>
    public IObservable<Generator> Process<T>(IObservable<T> source)
    {
        return source.Select(value => new Generator(Seed, Device));
    }
}
