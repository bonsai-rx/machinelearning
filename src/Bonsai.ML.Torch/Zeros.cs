using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Creates a tensor filled with zeros.
    /// </summary>
    [Combinator]
    [Description("Creates a tensor filled with zeros.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class Zeros
    {
        /// <summary>
        /// The size of the tensor.
        /// </summary>
        public long[] Size { get; set; } = [0];

        /// <summary>
        /// Generates an observable sequence of tensors filled with zeros.
        /// </summary>
        /// <returns></returns>
        public IObservable<Tensor> Process()
        {
            return Observable.Defer(() => Observable.Return(ones(Size)));
        }

        /// <summary>
        /// Generates an observable sequence of tensors filled with zeros for each element of the input sequence.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process<T>(IObservable<T> source)
        {
            return source.Select(value => {
                return ones(Size);
            });
        }
    }
}