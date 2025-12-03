using System;
using System.Linq;
using System.ComponentModel;
using System.Reactive;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;
using System.Threading.Tasks;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that performs a single optimization step using the specified optimizer.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Performs a single optimization step.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class OptimizationStep
{
    /// <summary>
    /// Performs a single optimization step using the specified optimizer.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<optim.Optimizer> Process(IObservable<optim.Optimizer> source)
    {
        return source.Do(input => input.step());
    }
}