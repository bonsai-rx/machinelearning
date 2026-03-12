using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes the QR decomposition of a matrix.
/// </summary>
[Combinator]
[Description("Computes the QR decomposition of a matrix.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class QRDecomposition
{
    /// <summary>
    /// Gets or sets the mode of the QR decomposition.
    /// </summary>
    [Description("The mode of the QR decomposition.")]
    public QRMode Mode { get; set; } = QRMode.Reduced;

    /// <summary>
    /// Computes the QR decomposition of a matrix.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<QRDecompositionResult> Process(IObservable<Tensor> source)
    {
        return source.Select(tensor => new QRDecompositionResult(qr(tensor, mode: Mode)));
    }

    /// <summary>
    /// Represents the result of a QR decomposition.
    /// </summary>
    /// <param name="result"></param>
    public readonly struct QRDecompositionResult((Tensor Q, Tensor R) result)
    {
        /// <summary>
        /// Gets the orthogonal matrix Q.
        /// </summary>
        public Tensor Q => result.Q;

        /// <summary>
        /// Gets the upper triangular matrix R.
        /// </summary>
        public Tensor R => result.R;
    }

}
