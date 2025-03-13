using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra
{
    /// <summary>
    /// Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.
    /// </summary>
    [Combinator]
    [Description("Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class CholeskyDecomposition
    {
        /// <summary>
        /// Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(linalg.cholesky);
        }
    }
}