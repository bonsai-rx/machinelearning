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
    /// Whether to compute the full or reduced SVD.
    /// </summary>
    [Description("Whether to compute the full or reduced SVD.")]
    public bool FullMatrices { get; set; } = false;

    /// <summary>
    /// Computes the singular value decomposition (SVD) of a matrix.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<SingularValueDecompositionResult> Process(IObservable<Tensor> source)
    {
        return source.Select(tensor => new SingularValueDecompositionResult(linalg.svd(tensor, fullMatrices: FullMatrices)));
    }

    /// <summary>
    /// Represents the result of a singular value decomposition.
    /// </summary>
    /// <param name="result"></param>
    public readonly struct SingularValueDecompositionResult((
        Tensor u,
        Tensor s,
        Tensor vh
    ) result)
    {
        /// <summary>
        /// The U tensor.
        /// </summary>
        public Tensor U => result.u;

        /// <summary>
        /// The singular values.
        /// </summary>
        public Tensor S => result.s;

        /// <summary>
        /// The Vh tensor.
        /// </summary>
        public Tensor Vh => result.vh;
    }
}
