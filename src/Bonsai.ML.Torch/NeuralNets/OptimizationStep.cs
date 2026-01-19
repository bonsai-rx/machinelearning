using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Collections.Generic;
using System.Linq.Expressions;
using Bonsai.Expressions;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that performs a single optimization step using the specified optimizer.
/// </summary>
[Combinator]
[Description("Performs a single optimization step.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class OptimizationStep
{
    /// <summary>
    /// Performs a single step using the specified optimizer.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<T> Process<T>(IObservable<T> source) where T : optim.Optimizer
    {
        return source.Do(input => input.step());
    }

    /// <summary>
    /// Performs a single step using the specified learning rate scheduler.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LRScheduler> Process(IObservable<LRScheduler> source)
    {
        return source.Do(input => input.step());
    }
}
