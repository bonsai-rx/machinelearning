using Bonsai;
using static TorchSharp.torch;
using TorchSharp;
using System;
using System.Reactive.Linq;
using System.Xml.Serialization;
using System.ComponentModel;

namespace Bonsai.ML.Torch.Random;

/// <summary>
/// Represents an operator that creates a specific instance of a random number generator (RNG) with the specified seed and device.
/// </summary>
[Combinator]
[Description("Creates a specific instance of a random number generator (RNG) with the specified seed and device.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CreateGenerator
{
    /// <summary>
    /// The device on which to create the generator.
    /// </summary>
    [XmlIgnore]
    [Description("The device on which to create the generator.")]
    public Device Device { get; set; }

    /// <summary>
    /// The seed for the random number generator.
    /// </summary>
    [Description("The seed for the random number generator.")]
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
    /// <typeparam name="T">The type of the elements in the input sequence.</typeparam>
    /// <param name="source">The input observable sequence.</param>
    /// <returns>An observable sequence of random number generators.</returns>
    public IObservable<Generator> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => new Generator(Seed, Device));
    }
}
