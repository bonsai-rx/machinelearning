using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Index;

/// <summary>
/// Represents an index that is created from a tensor.
/// </summary>
[Combinator]
[Description("Represents an index that is created from a tensor.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class TensorIndex
{   
    /// <summary>
    /// Converts the input tensor into a tensor index.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.torch.TensorIndex> Process(IObservable<Tensor> source)
    {
        return source.Select(TorchSharp.torch.TensorIndex.Tensor);
    }
}