using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Collections.Generic;
using static TorchSharp.torch;
using System.Linq;

namespace Bonsai.ML.Tensors
{
    /// <summary>
    /// Creates a mesh grid from an observable sequence of enumerable of 1-D tensors.
    /// </summary>
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class MeshGrid
    {
        /// <summary>
        /// The indexing mode to use for the mesh grid.
        /// </summary>
        public string Indexing { get; set; } = "ij";

        /// <summary>
        /// Creates a mesh grid from the input tensors.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor[]> Process(IObservable<IEnumerable<Tensor>> source)
        {
            return source.Select(tensors => meshgrid(tensors, indexing: Indexing));
        }
    }
}