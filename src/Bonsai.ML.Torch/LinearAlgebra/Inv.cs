using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Torch.LinearAlgebra
{
    /// <summary>
    /// Computes the inverse of the input matrix.
    /// </summary>
    [Combinator]
    [Description("Computes the inverse of the input matrix.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Inv
    {
        /// <summary>
        /// Computes the inverse of the input matrix.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(inv);
        }
    }
}