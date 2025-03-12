using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Takes the mean of the tensor along the specified dimensions.
    /// </summary>
    [Combinator]
    [Description("Takes the mean of the tensor along the specified dimensions.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Mean
    {
        /// <summary>
        /// The dimensions along which to compute the mean.
        /// </summary>
        [Description("The dimensions along which to compute the mean.")]
        [TypeConverter(typeof(UnidimensionalArrayConverter))]
        public long[] Dimensions { get; set; }

        /// <summary>
        /// Takes the mean of the tensor along the specified dimensions.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(input => input.mean(Dimensions));
        }
    }
}