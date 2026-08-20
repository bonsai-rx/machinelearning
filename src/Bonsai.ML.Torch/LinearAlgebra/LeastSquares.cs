using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the solution to the least squares and least norm problems for a full rank
/// matrix A of size m*n and a matrix B of size m*k.
/// </summary>
[Combinator]
[Description("Computes the solution to the least squares and least norm problems for a full rank matrix A of size m*n and a matrix B of size m*k.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class LeastSquares
{
    /// <summary>
    /// Computes the solution to the least squares and least norm problems for a full rank matrix A of size m*n and a
    /// matrix B of size m*k.
    /// </summary>
    public IObservable<LeastSquaresResult> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(value => new LeastSquaresResult(linalg.lstsq(value.Item1, value.Item2)));
    }
}
