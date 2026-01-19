using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that computes backward on the input tensor.
/// </summary>
[Combinator]
[Description("Computes backward on the input tensor.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class Backward
{
    /// <summary>
    /// Computes backward on the input tensor.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Do(input => input.backward());
    }
}