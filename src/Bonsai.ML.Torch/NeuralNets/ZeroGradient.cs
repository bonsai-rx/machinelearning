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
/// Represents an operator that sets the gradient of the specified optimizer to zero.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Sets the gradient to zero.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class ZeroGradient
{
    /// <summary>
    /// Sets the gradient of the specified optimizer to zero.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<optim.Optimizer> Process(IObservable<optim.Optimizer> source)
    {
        return source.Do(input => input.zero_grad());
    }
}