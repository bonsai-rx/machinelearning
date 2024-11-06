using static TorchSharp.torch;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace Bonsai.ML.Torch
{
    [Combinator]
    [Description("Constructs a tensor by repeating the elements of input. The Dimensions argument specifies the number of repetitions in each dimension.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Tile
    {
        public long[] Dimensions { get; set; }

        public IObservable<Tensor> Process(IObservable<Tensor> source)
        {
            return source.Select(tensor => {
                return tile(tensor, Dimensions);
            });
        }
    }
}
