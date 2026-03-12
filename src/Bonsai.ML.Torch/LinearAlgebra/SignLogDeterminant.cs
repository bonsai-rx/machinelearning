using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the sign and natural logarithm of the absolute value of the determinant of a square matrix.
/// </summary>
[Combinator]
[Description("Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class SignLogDeterminant
{
    /// <summary>
    /// Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<SignLogDeterminantResult> Process(IObservable<Tensor> source)
    {
        return source.Select(result => new SignLogDeterminantResult(linalg.slogdet(result)));
    }

    /// <summary>
    /// Represents the result of computing the sign and natural logarithm of the absolute value of the determinant.
    /// </summary>
    /// <param name="result"></param>
    public readonly struct SignLogDeterminantResult((Tensor sign, Tensor logabsdet) result)
    {
        /// <summary>
        /// Gets the sign of the determinant.
        /// </summary>
        public Tensor Sign => result.sign;

        /// <summary>
        /// Gets the natural logarithm of the absolute value of the determinant.
        /// </summary>
        public Tensor LogAbsDeterminant => result.logabsdet;
    }
}
