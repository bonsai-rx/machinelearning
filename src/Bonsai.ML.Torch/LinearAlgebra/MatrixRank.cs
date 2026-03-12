using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the numerical rank of a matrix.
/// </summary>
[Combinator]
[Description("Computes the numerical rank of a matrix.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class MatrixRank
{
    /// <summary>
    /// Gets or sets the absolute tolerance for singular values to be considered non-zero.
    /// </summary>
    [Description("The absolute tolerance for singular values to be considered non-zero.")]
    public double? AbsoluteTolerance { get; set; } = null;

    /// <summary>
    /// Gets or sets the relative tolerance for singular values to be considered non-zero.
    /// </summary>
    [Description("The relative tolerance for singular values to be considered non-zero.")]
    public double? RelativeTolerance { get; set; } = null;

    /// <summary>
    /// Gets or sets a value indicating whether to treat the input matrix as Hermitian if input is complex or symmetric if real.
    /// </summary>
    [Description("Indicates whether to treat the input matrix as Hermitian if input is complex or symmetric if real.")]
    public bool Hermitian { get; set; } = false;

    /// <summary>
    /// Computes the numerical rank of a matrix.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(input => matrix_rank(input, atol: AbsoluteTolerance, rtol: RelativeTolerance, hermitian: Hermitian));
    }
}
