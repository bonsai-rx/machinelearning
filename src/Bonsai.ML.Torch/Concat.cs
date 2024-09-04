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
    [WorkflowElementCategory(ElementCategory.Combinator)]
    public class Concat
    {
        /// <summary>
        /// The dimension along which to concatenate the tensors.
        /// </summary>
        public long Dimension { get; set; } = 0;

        /// <summary>
        /// Takes any number of observable sequences of tensors and concatenates the input tensors along the specified dimension by zipping each tensor together.
        /// </summary>
        public IObservable<Tensor> Process(params IObservable<Tensor>[] sources)
        {
            return sources.Aggregate((current, next) =>
                current.Zip(next, (tensor1, tensor2) =>
                    cat([tensor1, tensor2], Dimension)));
        }

        /// <summary>
        /// Concatenates the input tensors along the specified dimension.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor>> source)
        {
            return source.Select(value =>
            {
                return cat([value.Item1, value.Item2], Dimension);
            });
        }

        /// <summary>
        /// Concatenates the input tensors along the specified dimension.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor>> source)
        {
            return source.Select(value =>
            {
                return cat([value.Item1, value.Item2, value.Item3], Dimension);
            });
        }

        /// <summary>
        /// Concatenates the input tensors along the specified dimension.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor>> source)
        {
            return source.Select(value =>
            {
                return cat([value.Item1, value.Item2, value.Item3, value.Item4], Dimension);
            });
        }

        /// <summary>
        /// Concatenates the input tensors along the specified dimension.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor>> source)
        {
            return source.Select(value =>
            {
                return cat([value.Item1, value.Item2, value.Item3, value.Item4, value.Item5], Dimension);
            });
        }

        /// <summary>
        /// Concatenates the input tensors along the specified dimension.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> source)
        {
            return source.Select(value =>
            {
                return cat([value.Item1, value.Item2, value.Item3, value.Item4, value.Item5, value.Item6], Dimension);
            });
        }

        /// <summary>
        /// Concatenates the input tensors along the specified dimension.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<IEnumerable<Tensor>> source)
        {
            return source.Select(value =>
            {
                return cat(value.ToList(), Dimension);
            });
        }
    }
}
