using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra
{
    /// <summary>
    /// Computes the determinant of a square matrix.
    /// </summary>
    [Combinator]
    [Description("Computes the determinant of a square matrix.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Det
    {
        /// <summary>
        /// Computes the determinant of a square matrix.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(linalg.det);
        }
    }
}