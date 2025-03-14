using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Creates an empty tensor with the given data type and size.
    /// </summary>
    [Combinator]
    [Description("Creates an empty tensor with the given data type and size.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class Empty
    {
        /// <summary>
        /// The size of the tensor.
        /// </summary>
        [Description("The size of the tensor.")]
        [TypeConverter(typeof(UnidimensionalArrayConverter))]
        public long[] Size { get; set; } = [0];

        /// <summary>
        /// The data type of the tensor elements.
        /// </summary>
        [Description("The data type of the tensor elements.")]
        public ScalarType? Type { get; set; } = null;

        /// <summary>
        /// The device on which to create the tensor.
        /// </summary>
        [Description("The device on which to create the tensor.")]
        [XmlIgnore]
        public Device Device { get; set; } = null;

        /// <summary>
        /// Creates an empty tensor with the given data type and size.
        /// </summary>
        public IObservable<Tensor> Process()
        {
            return Observable.Return(empty(Size, dtype: Type, device: Device));
        }

        /// <summary>
        /// Generates an observable sequence of empty tensors for each element of the input sequence.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process<T>(IObservable<T> source)
        {
            return source.Select(value => empty(Size, dtype: Type, device: Device));
        }
    }
}
