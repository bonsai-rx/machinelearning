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
    public class SingularValueDecomposition
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
        public IObservable<SvdResult> Process(IObservable<Tensor> source)
        {
            return source.Select(tensor =>
            {
                var (U, S, Vh) = linalg.svd(tensor, fullMatrices: FullMatrices);
                return new SvdResult(U, S, Vh);
            });
        }

        /// <summary>
        /// Represents the result of a singular value decomposition.
        /// </summary>
        /// <param name="u"></param>
        /// <param name="s"></param>
        /// <param name="vh"></param>
        public readonly struct SvdResult(
            Tensor u,
            Tensor s,
            Tensor vh)
        {
            /// <summary>
            /// The U tensor.
            /// </summary>
            public Tensor U => u;

            /// <summary>
            /// The singular values.
            /// </summary>
            public Tensor S => s;

            /// <summary>
            /// The Vh tensor.
            /// </summary>
            public Tensor Vh => vh;
        }
    }
}