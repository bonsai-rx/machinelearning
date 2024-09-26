using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Returns an empty tensor with the given data type and size.
    /// </summary>
    [Combinator]
    [Description("Converts the input tensor into an OpenCV mat.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Empty
    {

        /// <summary>
        /// The size of the tensor.
        /// </summary>
        public long[] Size { get; set; } = [0];

        /// <summary>
        /// The data type of the tensor elements.
        /// </summary>
        public ScalarType Type { get; set; } = ScalarType.Float32;

        /// <summary>
        /// Returns an empty tensor with the given data type and size.
        /// </summary>
        public IObservable<Tensor> Process()
        {
            return Observable.Defer(() =>
            {
                return Observable.Return(empty(Size, Type));
            });
        }
    }
}
