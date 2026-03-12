using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the eigenvalue decomposition of a square matrix if it exists.
/// </summary>
[Combinator]
[Description("Computes the eigenvalue decomposition of a square matrix if it exists.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class EigenvalueDecomposition
{
    /// <summary>
    /// Computes the eigenvalue decomposition of a square matrix if it exists.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<EigenDecompositionResult> Process(IObservable<Tensor> source)
    {
        return source.Select(tensor => new EigenDecompositionResult(linalg.eig(tensor)));
    }

    /// <summary>
    /// Represents the result of an eigenvalue decomposition.
    /// </summary>
    /// <param name="result">The tuple containing the eigenvalues and eigenvectors.</param>
    public readonly struct EigenDecompositionResult((Tensor eigenvalues, Tensor eigenvectors) result)
    {
        /// <summary>
        /// Gets the eigenvalues of the decomposition.
        /// </summary>
        public Tensor Eigenvalues => result.eigenvalues;

        /// <summary>
        /// Gets the eigenvectors of the decomposition.
        /// </summary>
        public Tensor Eigenvectors => result.eigenvectors;
    }
}
