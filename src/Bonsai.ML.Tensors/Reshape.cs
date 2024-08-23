using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Tensors
{
    /// <summary>
    /// Reshapes the input tensor according to the specified dimensions.
    /// </summary>
    [Combinator]
    [Description("Reshapes the input tensor according to the specified dimensions.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Reshape
    {
        /// <summary>
        /// The dimensions of the reshaped tensor.
        /// </summary>
        public long[] Dimensions { get; set; } = [0];

        /// <summary>
        /// Reshapes the input tensor according to the specified dimensions.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(input => input.reshape(Dimensions));
        }
    }
}