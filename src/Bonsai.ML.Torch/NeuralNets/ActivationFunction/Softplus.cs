using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a softplus activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html"/> for more information.
/// </remarks>
[Description("Creates a softplus module.")]
public class Softplus
{
    /// <summary>
    /// The beta value for the softplus formula.
    /// </summary>
    [Description("The beta value for the softplus formula.")]
    public double Beta { get; set; } = 1D;

    /// <summary>
    /// The threshold value for which values above it use a linear function.
    /// </summary>
    [Description("The threshold value for which values above it use a linear function.")]
    public double Threshold { get; set; } = 20D;

    /// <summary>
    /// Creates a Softplus module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Softplus> Process()
    {
        return Observable.Return(Softplus(Beta, Threshold));
    }

    /// <summary>
    /// Creates a Softplus module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Softplus> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Softplus(Beta, Threshold));
    }
}
