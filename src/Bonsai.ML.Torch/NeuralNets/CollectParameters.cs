using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Linq;
using System.Collections.Generic;
using TorchSharp.Modules;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that collects the parameters of torch modules into a collection.
/// </summary>
[Combinator]
[Description("Collects the parameters from torch modules into a collection.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class CollectParameters
{
    /// <summary>
    /// Collects the parameters from torch modules into a collection.
    /// </summary>
    /// <param name="sources"></param>
    /// <returns></returns>
    public IObservable<IEnumerable<Parameter>> Process(params IObservable<Module>[] sources)
    {
        return Observable
            .Concat(sources.Select(source =>
                source.Take(1)))
            .SelectMany(module =>
            {
                return module.parameters(recurse: true);
            }).ToList();
    }
}