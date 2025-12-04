using Bonsai.Expressions;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Linq;
using System.Linq.Expressions;
using System.Collections.Generic;
using System.Reflection;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

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
    public IObservable<IEnumerable<Parameter>> Process(params IObservable<nn.Module>[] sources)
    {
        return Observable.Concat(sources)
            .SelectMany(module =>
            {
                return module.parameters(recurse: true);
            }).ToList();
    }
}