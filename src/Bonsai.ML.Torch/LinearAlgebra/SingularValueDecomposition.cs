using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the singular value decomposition (SVD) of a matrix.
/// </summary>
[Combinator]
[Description("Computes the singular value decomposition (SVD) of a matrix.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class SingularValueDecomposition
{
    /// <summary>
    /// Gets or sets whether to compute the full or reduced SVD.
    /// </summary>
    /// <remarks>
    /// True indicates that the full SVD is computed and the vector matrices U and Vh may be padded with extra columns/
    /// rows to make them square and unitary; otherwise, the reduced SVD is computed and the vector matrices U and Vh
    /// have only the minimum number of columns/rows.
    /// </remarks>
    [Description("True indicates that the full SVD is computed and the vector matrices U and Vh have extra columns/rows to make them unitary; otherwise, the reduced SVD is computed and the vector matrices U and Vh have the minimum number of columns/rows.")]
    public bool FullMatrices { get; set; } = true;

    /// <summary>
    /// Computes the singular value decomposition (SVD) of a matrix.
    /// </summary>
    public IObservable<SingularValueDecompositionResult> Process(IObservable<Tensor> source)
    {
        return source.Select(tensor => new SingularValueDecompositionResult(linalg.svd(tensor, fullMatrices: FullMatrices)));
    }
}
