using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Tensors
{
    /// <summary>
    /// Creates a 1-D tensor of linearly interpolated values within a given range given the start, end, and count.
    /// </summary>
    [Combinator]
    [Description("Creates a 1-D tensor of linearly interpolated values within a given range given the start, end, and count.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class Linspace
    {
        /// <summary>
        /// The start of the range.
        /// </summary>
        public int Start { get; set; } = 0;

        /// <summary>
        /// The end of the range.
        /// </summary>
        public int End { get; set; } = 1;

        /// <summary>
        /// The number of points to generate.
        /// </summary>
        public int Count { get; set; } = 10;

        /// <summary>
        /// Generates an observable sequence of 1-D tensors created with the <see cref="linspace"/> function.
        /// </summary>
        /// <returns></returns>
        public IObservable<Tensor> Process()
        {
            return Observable.Defer(() => Observable.Return(linspace(Start, End, Count)));
        }
    }
}