using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Represents an operator that creates a Kullback-Leibler divergence loss module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html"/> for more information.
/// </remarks>
[Description("Creates a Kullback-Leibler divergence loss module.")]
[DisplayName("KLDiv")]
public class KullbackLeiblerDivergence
{
    /// <summary>
    /// Determines whether the target is given as log-probabilities.
    /// </summary>
    [Description("Determines whether the target is given as log-probabilities.")]
    public bool LogTarget { get; set; } = true;

    /// <summary>
    /// The reduction type to apply to the output.
    /// </summary>
    [Description("The reduction type to apply to the output.")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Creates a Kullback-Leibler divergence loss module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.KLDivLoss> Process()
    {
        return Observable.Return(KLDivLoss(LogTarget, Reduction));
    }

    /// <summary>
    /// Creates a Kullback-Leibler divergence loss module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.KLDivLoss> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => KLDivLoss(LogTarget, Reduction));
    }
}
