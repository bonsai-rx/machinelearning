using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Creates a 1-D tensor of equally spaced values within a given range given the start, end, and count.
    /// </summary>
    [Combinator]
    [ResetCombinator]
    [Description("Creates a 1-D tensor of equally spaced values within a given range given the start, end, and count.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class LinSpace
    {
        /// <summary>
        /// The start of the range.
        /// </summary>
        [Description("The start of the range.")]
        public int Start { get; set; } = 0;

        /// <summary>
        /// The end of the range.
        /// </summary>
        [Description("The end of the range.")]
        public int End { get; set; } = 1;

        /// <summary>
        /// The number of points to generate.
        /// </summary>
        [Description("The number of points to generate.")]
        public int Count { get; set; } = 10;

        /// <summary>
        /// The device on which to create the tensor.
        /// </summary>
        [Description("The device on which to create the tensor.")]
        [XmlIgnore]
        public Device Device { get; set; } = null;

        /// <summary>
        /// The data type of the tensor.
        /// </summary>
        [Description("The data type of the tensor.")]
        public ScalarType? Type { get; set; } = null;

        /// <summary>
        /// Generates an observable sequence of 1-D tensors created with the <see cref="linspace"/> function.
        /// </summary>
        /// <returns></returns>
        public IObservable<Tensor> Process()
        {
            return Observable.Return(linspace(Start, End, Count, dtype: Type, device: Device));
        }

        /// <summary>
        /// Generates an observable sequence of 1-D tensors created with the <see cref="linspace"/> function for each element of the input sequence.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process<T>(IObservable<T> source)
        {
            return source.Select(value => linspace(Start, End, Count, dtype: Type, device: Device));
        }
    }
}