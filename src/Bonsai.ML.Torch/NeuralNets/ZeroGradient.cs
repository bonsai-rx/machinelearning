using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that sets the gradient of the input optimizer to zero.
/// </summary>
[Combinator]
[Description("Sets the gradient of the input optimizer to zero.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class ZeroGradient
{
    /// <summary>
    /// Sets the gradient of the input optimizer to zero.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<T> Process<T>(IObservable<T> source) where T : optim.Optimizer
    {
        return source.Do(input => input.zero_grad());
    }
}
