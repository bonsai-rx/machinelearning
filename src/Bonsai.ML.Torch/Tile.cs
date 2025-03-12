using static TorchSharp.torch;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Constructs a tensor by repeating the elements of input.
    /// </summary>
    [Combinator]
    [Description("Constructs a tensor by repeating the elements of input.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Tile
    {
        /// <summary>
        /// The number of repetitions in each dimension.
        /// </summary>
        [Description("The number of repetitions in each dimension.")]
        [TypeConverter(typeof(UnidimensionalArrayConverter))]
        public long[] Dimensions { get; set; }

        /// <summary>
        /// Constructs a tensor by repeating the elements of input along the specified dimensions.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(tensor => {
                return tile(tensor, Dimensions);
            });
        }
    }
}
