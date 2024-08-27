using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Tensors
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
    }
}