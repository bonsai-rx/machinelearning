using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Normalization;

/// <summary>
/// Represents an operator that creates a local response normalization module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.LocalResponseNorm.html"/> for more information.
/// </remarks>
[Description("Creates a local response normalization module.")]
public class LocalResponseNorm
{
    /// <summary>
    /// The number of neighboring channels used for normalization.
    /// </summary>
    [Description("The number of neighboring channels used for normalization.")]
    public long Size { get; set; }

    /// <summary>
    /// The alpha parameter, which serves as a scaling factor.
    /// </summary>
    [Description("The alpha parameter, which serves as a scaling factor.")]
    public double Alpha { get; set; } = 0.0001D;

    /// <summary>
    /// The beta parameter, which serves as an exponent.
    /// </summary>
    [Description("The beta parameter, which serves as an exponent.")]
    public double Beta { get; set; } = 0.75D;

    /// <summary>
    /// The k parameter, which serves as an additive constant.
    /// </summary>
    [Description("The k parameter, which serves as an additive constant.")]
    public double K { get; set; } = 1D;

    /// <summary>
    /// Creates a LocalResponseNorm module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(LocalResponseNorm(Size, Alpha, Beta, K));
    }

    /// <summary>
    /// Creates a LocalResponseNorm module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => LocalResponseNorm(Size, Alpha, Beta, K));
    }
}
