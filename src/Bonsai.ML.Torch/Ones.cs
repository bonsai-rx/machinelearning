using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Creates a tensor filled with ones.
    /// </summary>
    [Combinator]
    [ResetCombinator]
    [Description("Creates a tensor filled with ones.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class Ones
    {
        /// <summary>
        /// The size of the tensor.
        /// </summary>
        [Description("The size of the tensor.")]
        [TypeConverter(typeof(UnidimensionalArrayConverter))]
        public long[] Size { get; set; } = [0];

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
        /// Generates an observable sequence of tensors filled with ones.
        /// </summary>
        /// <returns></returns>
        public IObservable<Tensor> Process()
        {
            return Observable.Return(ones(Size, dtype: Type, device: Device));
        }

        /// <summary>
        /// Generates an observable sequence of tensors filled with ones for each element of the input sequence.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process<T>(IObservable<T> source)
        {
            return source.Select(value => ones(Size, dtype: Type, device: Device));
        }
    }
}