using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Creates a new view of the input tensor with the specified dimensions.
    /// </summary>
    [Combinator]
    [Description("Creates a new view of the input tensor with the specified dimensions.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class View
    {
        /// <summary>
        /// The dimensions of the reshaped tensor.
        /// </summary>
        [Description("The dimensions of the reshaped tensor.")]
        [TypeConverter(typeof(UnidimensionalArrayConverter))]
        public long[] Dimensions { get; set; } = [0];

        /// <summary>
        /// Creates a new view of the input tensor with the specified dimensions.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(input => input.view(Dimensions));
        }
    }
}