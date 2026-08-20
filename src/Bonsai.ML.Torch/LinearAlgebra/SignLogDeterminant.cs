using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the sign and natural logarithm of the absolute value of the determinant of a
/// square matrix.
/// </summary>
[Combinator]
[Description("Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class SignLogDeterminant
{
    /// <summary>
    /// Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix.
    /// </summary>
    public IObservable<SignLogDeterminantResult> Process(IObservable<Tensor> source)
    {
        return source.Select(tensor => new SignLogDeterminantResult(linalg.slogdet(tensor)));
    }
}
