using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the inverse of the input matrix.
/// </summary>
[Combinator]
[Description("Computes the inverse of the input matrix.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Inverse
{
    /// <summary>
    /// Computes the inverse of the input matrix.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(inv);
    }
}
