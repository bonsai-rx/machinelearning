using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra
{
    /// <summary>
    /// Computes a vector or matrix norm.
    /// </summary>
    [Combinator]
    [Description("Computes a vector or matrix norm.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Norm
    {
        /// <summary>
        /// The dimensions along which to compute the norm.
        /// </summary>
        [TypeConverter(typeof(UnidimensionalArrayConverter))]
        public long[] Dimensions { get; set; } = null;

        /// <summary>
        /// If true, the reduced dimensions are retained in the result as dimensions with size one.
        /// </summary>
        public bool Keepdim { get; set; } = false;

        /// <summary>
        /// Computes a matrix norm.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(tensor => linalg.norm(tensor, dims: Dimensions, keepdim: Keepdim));
        }
    }
}