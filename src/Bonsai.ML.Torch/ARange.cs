using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using TorchSharp;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Creates a 1-D tensor of values within a given range given the start, end, and step.
    /// </summary>
    [Combinator]
    [Description("Creates a 1-D tensor of values within a given range given the start, end, and step.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class ARange
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
        public int End { get; set; } = 10;

        /// <summary>
        /// The step size between values.
        /// </summary>
        [Description("The step size between values.")]
        public int Step { get; set; } = 1;

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
        /// Generates an observable sequence of 1-D tensors created with the <see cref="arange(Scalar, Scalar, Scalar, ScalarType?, Device?, bool)"/> function.
        /// </summary>
        public IObservable<Tensor> Process()
        {
            return Observable.Return(arange(Start, End, Step, dtype: Type, device: Device));
        }

        /// <summary>
        /// Generates an observable sequence of 1-D tensors created with the <see cref="arange(Scalar, Scalar, Scalar, ScalarType?, Device?, bool)"/> function for each element of the input sequence.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process<T>(IObservable<T> source)
        {
            return source.Select(value => arange(Start, End, Step, dtype: Type, device: Device));
        }
    }
}
