using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Moves the input tensor to the specified device.
    /// </summary>
    [Combinator]
    [Description("Moves the input tensor to the specified device.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ToDevice
    {
        /// <summary>
        /// The device to which the input tensor should be moved.
        /// </summary>
        public Device Device { get; set; }

        /// <summary>
        /// Returns the input tensor moved to the specified device.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(tensor => {
                return tensor.to(Device);
            });
        }
    }
}