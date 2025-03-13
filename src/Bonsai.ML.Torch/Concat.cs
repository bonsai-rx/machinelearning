using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Collections.Generic;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Concatenates tensors along a given dimension.
    /// </summary>
    [Combinator]
    [Description("Concatenates tensors along a given dimension.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Concat
    {
        /// <summary>
        /// The dimension along which to concatenate the tensors.
        /// </summary>
        [Description("The dimension along which to concatenate the tensors.")]
        public long Dimension { get; set; } = 0;

        /// <summary>
        /// Concatenates the input tensors along the specified dimension.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor>> source)
        {
            return source.Select(value => cat([value.Item1, value.Item2], Dimension));
        }

        /// <summary>
        /// Concatenates the input tensors along the specified dimension.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor>> source)
        {
            return source.Select(value => cat([value.Item1, value.Item2, value.Item3], Dimension));
        }

        /// <summary>
        /// Concatenates the input tensors along the specified dimension.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor>> source)
        {
            return source.Select(value => cat([value.Item1, value.Item2, value.Item3, value.Item4], Dimension));
        }

        /// <summary>
        /// Concatenates the input tensors along the specified dimension.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor>> source)
        {
            return source.Select(value => cat([value.Item1, value.Item2, value.Item3, value.Item4, value.Item5], Dimension));
        }

        /// <summary>
        /// Concatenates the input tensors along the specified dimension.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> source)
        {
            return source.Select(value => cat([value.Item1, value.Item2, value.Item3, value.Item4, value.Item5, value.Item6], Dimension));
        }

        /// <summary>
        /// Concatenates the input tensors along the specified dimension.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> source)
        {
            return source.Select(value => cat([value.Item1, value.Item2, value.Item3, value.Item4, value.Item5, value.Item6, value.Item7], Dimension));
        }

        /// <summary>
        /// Concatenates the input tensors along the specified dimension.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<IEnumerable<Tensor>> source)
        {
            return source.Select(value => cat([.. value], Dimension));
        }
    }
}
