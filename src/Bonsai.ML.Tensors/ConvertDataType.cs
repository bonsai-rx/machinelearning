using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Tensors
{
    /// <summary>
    /// Converts the input tensor to the specified scalar type.
    /// </summary>
    [Combinator]
    [Description("Converts the input tensor to the specified scalar type.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ConvertDataType
    {
        /// <summary>
        /// The scalar type to which to convert the input tensor.
        /// </summary>
        public ScalarType Type { get; set; } = ScalarType.Float32;

        /// <summary>
        /// Returns an observable sequence that converts the input tensor to the specified scalar type.
        /// </summary>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(tensor =>
            {
                return tensor.to_type(Type);
            });
        }
    }
}
