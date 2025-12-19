using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Converts the data type of the input tensor to the newly specified scalar type.
    /// </summary>
    [Combinator]
    [Description("Converts the data type of the input tensor to the newly specified scalar type.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ConvertScalarType
    {
        /// <summary>
        /// The scalar type with which to convert the input tensor.
        /// </summary>
        [Description("The scalar type with which to convert the input tensor.")]
        public ScalarType Type { get; set; } = ScalarType.Float32;

        /// <summary>
        /// Converts the data type of the input tensor to the newly specified scalar type.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(tensor => tensor.to_type(Type));
        }
    }
}
