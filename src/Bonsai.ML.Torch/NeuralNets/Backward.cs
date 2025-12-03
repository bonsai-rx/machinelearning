using System;
using System.Linq;
using System.ComponentModel;
using System.Reactive;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;
using System.Threading.Tasks;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that computes backward on the input tensor.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Computes backward on the input tensor.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class Backward
{
    /// <summary>
    /// Computes backward on the input tensor.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Do(input => input.backward());
    }
}