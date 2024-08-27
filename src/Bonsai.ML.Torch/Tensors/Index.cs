using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Tensors
{
    /// <summary>
    /// Indexes a tensor with the specified indices. Indices are specified as a comma-separated values. 
    /// Currently supports Python-style slicing syntax. This includes numeric indices, None, slices, and ellipsis.
    /// </summary>
    [Combinator]
    [Description("Indexes a tensor with the specified indices. Indices are specified as a comma-separated values. Currently supports Python-style slicing syntax. This includes numeric indices, None, slices, and ellipsis.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Index
    {
        /// <summary>
        /// The indices to use for indexing the tensor.
        /// </summary>
        public string Indexes { get; set; } = string.Empty;

        /// <summary>
        /// Indexes the input tensor with the specified indices.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            var index = Helpers.IndexHelper.ParseString(Indexes);
            return source.Select(tensor => {
                return tensor.index(index);
            });
        }
    }
}
