using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Creates a diagonal matrix. If input is a 1D tensor, it creates a diagonal matrix with the elements of the tensor on the diagonal.
    /// If input is a 2D tensor, it returns the diagonal elements as a 1D tensor.
    /// </summary>
    [Combinator]
    [ResetCombinator]
    [Description("Creates a diagonal matrix with the given data type, size, and value.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class Diagonal
    {
        /// <summary>
        /// The input matrix.
        /// </summary>
        [Description("The input matrix.")]
        [TypeConverter(typeof(UnidimensionalArrayConverter))]
        public double[] Input { get; set; }

        /// <summary>
        /// The data type of the tensor elements.
        /// </summary>
        [Description("The data type of the tensor elements.")]
        public ScalarType? Type { get; set; }

        /// <summary>
        /// The device on which to create the tensor.
        /// </summary>
        [Description("The device on which to create the tensor.")]
        [XmlIgnore]
        public Device Device { get; set; } = null;

        /// <summary>
        /// The diagonal offset. Default is 0, which means the main diagonal.
        /// </summary>
        [Description("The diagonal offset. Default is 0, which means the main diagonal.")]
        public int Offset { get; set; } = 0;

        /// <summary>
        /// Creates a diagonal matrix.
        /// </summary>
        public IObservable<Tensor> Process()
        {
            var inputTensor = tensor(Input, dtype: Type, device: Device);
            return Observable.Return(diag(inputTensor, Offset));
        }

        /// <summary>
        /// Generates an observable sequence of tensors by extracting the diagonal of the input.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            var inputTensor = tensor(Input, dtype: Type, device: Device);
            return source.Select(value => diag(inputTensor, Offset));
        }

        /// <summary>
        /// Generates an observable sequence of tensors by extracting the diagonal of the input.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process<T>(IObservable<T> source)
        {
            var inputTensor = tensor(Input, dtype: Type, device: Device);
            return source.Select(value => diag(inputTensor, Offset));
        }
    }
}
