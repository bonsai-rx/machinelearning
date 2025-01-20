using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra
{
    /// <summary>
    /// Computes the eigenvalue decomposition of a square matrix if it exists.
    /// </summary>
    [Combinator]
    [Description("Computes the eigenvalue decomposition of a square matrix if it exists.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Eig
    {
        /// <summary>
        /// Computes the eigenvalue decomposition of a square matrix if it exists.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tuple<Tensor, Tensor>> Process(IObservable<Tensor> source)
        {
            return source.Select(tensor => {
                var (eigvals, eigvecs) = linalg.eig(tensor);
                return Tuple.Create(eigvals, eigvecs);
            });
        }
    }
}