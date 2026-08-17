using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Linq;
using System.Collections.Generic;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that merges derived modules into a collection of base module types.
/// </summary>
[Combinator]
[Description("Merges derived modules into a collection of base module types.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class MergeModules
{
    /// <summary>
    /// Merges the input sequences of modules and casts them to the base module class.
    /// </summary>
    /// <param name="sources"></param>
    /// <returns></returns>
    public IObservable<IEnumerable<Module<Tensor, Tensor>>> Process(params IObservable<Module>[] sources)
    {
        return Observable
            .Concat(sources.Select(source =>
                source.Take(1)))
            .OfType<Module<Tensor, Tensor>>()
            .ToList()
            .Select(modules => modules.AsEnumerable());
    }
}
