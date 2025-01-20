using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Sets the value of the input tensor at the specified index.
    /// </summary>
    [Combinator]
    [Description("Sets the value of the input tensor at the specified index.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Set
    {
        /// <summary>
        /// The index at which to set the value.
        /// </summary>
        [Description("The index at which to set the value.")]
        public string Index { get; set; } = string.Empty;

        /// <summary>
        /// The value to set at the specified index.
        /// </summary>
        [XmlIgnore]
        [Description("The value to set at the specified index.")]
        public Tensor Value { get; set; } = null;

        /// <summary>
        /// Returns an observable sequence that sets the value of the input tensor at the specified index.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(tensor => {
                var indexes = IndexHelper.Parse(Index);
                return tensor.index_put_(Value, indexes);
            });
        }
    }
}