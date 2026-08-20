using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

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
    public IObservable<EigenvalueDecompositionResult> Process(IObservable<Tensor> source)
    {
        return source.Select(tensor => new EigenvalueDecompositionResult(eig(tensor)));
    }
}
