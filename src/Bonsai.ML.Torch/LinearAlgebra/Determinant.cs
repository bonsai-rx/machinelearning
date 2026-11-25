using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the determinant of a square matrix.
/// </summary>
[Combinator]
[Description("Computes the determinant of a square matrix.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Determinant
{
    /// <summary>
    /// Computes the determinant of a square matrix.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(linalg.det);
    }
}
