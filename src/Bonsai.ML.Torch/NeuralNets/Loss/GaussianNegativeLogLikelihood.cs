using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Represents an operator that creates a gaussian negative log likelihood (GaussianNLL) loss module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html"/> for more information.
/// </remarks>
[Description("Creates a gaussian negative log likelihood (GaussianNLL) loss module.")]
[DisplayName("GaussianNLL")]
public class GaussianNegativeLogLikelihood
{
    /// <summary>
    /// Determines whether to include the constant term in the loss.
    /// </summary>
    [Description("Determines whether to include the constant term in the loss.")]
    public bool Full { get; set; } = false;

    /// <summary>
    /// The value used to clamp the variance for numerical stability.
    /// </summary>
    [Description("The value used to clamp the variance for numerical stability")]
    public float Eps { get; set; } = 1E-08F;

    /// <summary>
    /// The reduction type to apply to the output.
    /// </summary>
    [Description("The reduction type to apply to the output.")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Creates a gaussian negative log likelihood (GaussianNLL) loss module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.GaussianNLLLoss> Process()
    {
        return Observable.Return(GaussianNLLLoss(Full, Eps, Reduction));
    }

    /// <summary>
    /// Creates a gaussian negative log likelihood (GaussianNLL) loss module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.GaussianNLLLoss> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => GaussianNLLLoss(Full, Eps, Reduction));
    }
}
