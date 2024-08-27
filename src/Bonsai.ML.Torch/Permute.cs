using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Permutes the dimensions of the input tensor according to the specified permutation.
    /// </summary>
    [Combinator]
    [Description("Permutes the dimensions of the input tensor according to the specified permutation.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Permute
    {
        /// <summary>
        /// The permutation of the dimensions.
        /// </summary>
        public long[] Dimensions { get; set; } = [0];

        /// <summary>
        /// Returns an observable sequence that permutes the dimensions of the input tensor according to the specified permutation.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(tensor => {
                return tensor.permute(Dimensions);
            });
        }
    }
}