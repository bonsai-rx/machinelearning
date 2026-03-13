using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Represents an operator that creates a connectionist temporal classification (CTC) loss module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html"/> for more information.
/// </remarks>
[Description("Creates a connectionist temporal classification (CTC) loss module.")]
[DisplayName("CTC")]
public class ConnectionistTemporalClassification
{
    /// <summary>
    /// The blank label.
    /// </summary>
    [Description("The blank label.")]
    public long Blank { get; set; } = 0;

    /// <summary>
    /// Determines whether to zero infinite losses and the associated gradients.
    /// </summary>
    [Description("Determines whether to zero infinite losses and the associated gradients.")]
    public bool ZeroInfinity { get; set; } = false;

    /// <summary>
    /// The reduction type to apply to the output.
    /// </summary>
    [Description("The reduction type to apply to the output.")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// Creates a connectionist temporal classification (CTC) loss module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.CTCLoss> Process()
    {
        return Observable.Return(CTCLoss(Blank, ZeroInfinity, Reduction));
    }

    /// <summary>
    /// Creates a connectionist temporal classification (CTC) loss module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.CTCLoss> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => CTCLoss(Blank, ZeroInfinity, Reduction));
    }
}
