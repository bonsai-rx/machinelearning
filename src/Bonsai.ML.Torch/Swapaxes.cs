using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    [Combinator]
    [Description("Swaps the axes of the input tensor.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Swapaxes
    {

        /// <summary>
        /// The value of axis 1.
        /// </summary>
        public long Axis1 { get; set; } = 0;

        /// <summary>
        /// The value of axis 2.
        /// </summary>
        public long Axis2 { get; set; } = 1;

        /// <summary>
        /// Returns an observable sequence that sets the value of the input tensor at the specified index.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(tensor => {
                return swapaxes(tensor, Axis1, Axis2);
            });
        }
    }
}