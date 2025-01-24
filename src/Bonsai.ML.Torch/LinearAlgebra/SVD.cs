using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra
{
    /// <summary>
    /// Computes the singular value decomposition (SVD) of a matrix.
    /// </summary>
    [Combinator]
    [Description("Computes the singular value decomposition (SVD) of a matrix.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class SVD
    {
        /// <summary>
        /// Whether to compute the full or reduced SVD.
        /// </summary>
        public bool FullMatrices { get; set; } = false;

        /// <summary>
        /// Computes the singular value decomposition (SVD) of a matrix.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tuple<Tensor, Tensor, Tensor>> Process(IObservable<Tensor> source)
        {
            return source.Select(tensor => {
                var (u, s, v) = linalg.svd(tensor, fullMatrices: FullMatrices);
                return Tuple.Create(u, s, v);
            });
        }
    }
}