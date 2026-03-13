using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Represents an operator that creates a multi-class multi-classification margin loss (MultiLabelMarginLoss) module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelMarginLoss.html"/> for more information.
/// </remarks>
[Description("Creates a multi-class multi-classification margin loss (MultiLabelMarginLoss) module.")]
public class MultiLabelMargin
{
    /// <summary>
    /// The reduction type to apply to the output.
    /// </summary>
    [Description("The reduction type to apply to the output.")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Creates a multi-label margin loss (MultiLabelMarginLoss) module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.MultiLabelMarginLoss> Process()
    {
        return Observable.Return(MultiLabelMarginLoss(Reduction));
    }

    /// <summary>
    /// Creates a multi-label margin loss (MultiLabelMarginLoss) module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.MultiLabelMarginLoss> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => MultiLabelMarginLoss(Reduction));
    }
}
