using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Represents an operator that creates a margin ranking loss module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html"/> for more information.
/// </remarks>
[Description("Creates a margin ranking loss module.")]
public class MarginRanking
{
    /// <summary>
    /// The margin, a float value.
    /// </summary>
    [Description("The margin, a float value.")]
    public double Margin { get; set; } = 0D;

    /// <summary>
    /// The reduction type to apply to the output.
    /// </summary>
    [Description("The reduction type to apply to the output.")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Creates a margin ranking loss module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.MarginRankingLoss> Process()
    {
        return Observable.Return(MarginRankingLoss(Margin, Reduction));
    }

    /// <summary>
    /// Creates a margin ranking loss module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.MarginRankingLoss> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => MarginRankingLoss(Margin, Reduction));
    }
}
