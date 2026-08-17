using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Represents an operator that creates a poisson negative log likelihood (PoissonNLL) loss module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html"/> for more information.
/// </remarks>
[Description("Creates a poisson negative log likelihood (PoissonNLL) loss module.")]
[DisplayName("PoissonNLL")]
public class PoissonNegativeLogLikelihood
{
    /// <summary>
    /// Determines whether to take the logarithm of the input.
    /// </summary>
    [Description("Determines whether to take the logarithm of the input.")]
    public bool LogInput { get; set; } = true;

    /// <summary>
    /// Determines whether to compute the full loss.
    /// </summary>
    [Description("Determines whether to compute the full loss.")]
    public bool Full { get; set; } = false;

    /// <summary>
    /// The small value added to avoid evaluating log(0) when LogInput is false.
    /// </summary>
    [Description("The small value added to avoid evaluating log(0) when LogInput is false.")]
    public float Eps { get; set; } = 1E-08F;

    /// <summary>
    /// The reduction type to apply to the output.
    /// </summary>
    [Description("The reduction type to apply to the output.")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Creates a poisson negative log likelihood (PoissonNLL) loss module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.PoissonNLLLoss> Process()
    {
        return Observable.Return(PoissonNLLLoss(LogInput, Full, Eps, Reduction));
    }

    /// <summary>
    /// Creates a poisson negative log likelihood (PoissonNLL) loss module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.PoissonNLLLoss> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => PoissonNLLLoss(LogInput, Full, Eps, Reduction));
    }
}
