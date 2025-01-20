using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Computes the sum of the input tensor elements along the specified dimensions.
    /// </summary>
    [Combinator]
    [Description("Computes the sum of the input tensor elements along the specified dimensions.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Sum
    {
        /// <summary>
        /// The dimensions along which to compute the sum.
        /// </summary>
        public long[] Dimensions { get; set; }

        /// <summary>
        /// Computes the sum of the input tensor elements along the specified dimensions.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(input => input.sum(Dimensions));
        }
    }
}