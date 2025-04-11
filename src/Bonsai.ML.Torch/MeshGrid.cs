using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Collections.Generic;
using static TorchSharp.torch;
using System.Linq;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Creates a mesh grid from an observable sequence of enumerable of 1-D tensors.
    /// </summary>
    [Combinator]
    [Description("Creates a mesh grid from an observable sequence of enumerable of 1-D tensors.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class MeshGrid
    {
        /// <summary>
        /// The indexing mode to use for the mesh grid.
        /// </summary>
        [Description("The indexing mode to use for the mesh grid.")]
        public string Indexing { get; set; } = "ij";

        /// <summary>
        /// Creates a mesh grid from the input tensors.
        /// </summary>
        public IObservable<Tensor[]> Process(IObservable<Tuple<Tensor, Tensor>> source)
        {
            return source.Select(value => meshgrid([value.Item1, value.Item2], indexing: Indexing));
        }

        /// <summary>
        /// Creates a mesh grid from the input tensors.
        /// </summary>
        public IObservable<Tensor[]> Process(IObservable<Tuple<Tensor, Tensor, Tensor>> source)
        {
            return source.Select(value => meshgrid([value.Item1, value.Item2, value.Item3], indexing: Indexing));
        }

        /// <summary>
        /// Creates a mesh grid from the input tensors.
        /// </summary>
        public IObservable<Tensor[]> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor>> source)
        {
            return source.Select(value => meshgrid([value.Item1, value.Item2, value.Item3, value.Item4], indexing: Indexing));
        }

        /// <summary>
        /// Creates a mesh grid from the input tensors.
        /// </summary>
        public IObservable<Tensor[]> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor>> source)
        {
            return source.Select(value => meshgrid([value.Item1, value.Item2, value.Item3, value.Item4, value.Item5], indexing: Indexing));
        }

        /// <summary>
        /// Creates a mesh grid from the input tensors.
        /// </summary>
        public IObservable<Tensor[]> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> source)
        {
            return source.Select(value => meshgrid([value.Item1, value.Item2, value.Item3, value.Item4, value.Item5, value.Item6], indexing: Indexing));
        }

        /// <summary>
        /// Creates a mesh grid from the input tensors.
        /// </summary>
        public IObservable<Tensor[]> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>> source)
        {
            return source.Select(value => meshgrid([value.Item1, value.Item2, value.Item3, value.Item4, value.Item5, value.Item6, value.Item7], indexing: Indexing));
        }

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