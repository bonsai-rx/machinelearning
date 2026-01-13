using System;
using System.Reactive.Linq;
using System.ComponentModel;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Random;

/// <summary>
/// Represents an operator that sets the global random seed for TorchSharp and creates a random number generator (RNG) with the specified seed.
/// </summary>
[Combinator]
[Description("Sets the global random seed for TorchSharp and creates a random number generator (RNG) with the specified seed.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ManualSeed
{
    /// <summary>
    /// The seed for the random number generator.
    /// </summary>
    public long Seed { get; set; } = 0;

    /// <summary>
    /// Sets the global random seed and creates a random number generator (RNG).
    /// </summary>
    /// <returns></returns>
    public IObservable<Generator> Process()
    {
        return Observable.Return(manual_seed(Seed));
    }

    /// <summary>
    /// Generates an observable sequence where each element sets the global random seed and creates a random number generator (RNG).
    /// </summary>
    /// <typeparam name="T">The type of the elements in the input sequence.</typeparam>
    /// <param name="source">The input observable sequence.</param>
    /// <returns>An observable sequence of random number generators.</returns>
    public IObservable<Generator> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => manual_seed(Seed));
    }
}
