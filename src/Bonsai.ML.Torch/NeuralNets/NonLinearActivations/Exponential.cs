using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.NonLinearActivations;

/// <summary>
/// Represents an operator that creates an exponential linear unit (ELU) activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.ELU.html"/> for more information.
/// </remarks>
[Description("Creates an exponential linear unit (ELU) activation function.")]
public class Exponential
{
    /// <summary>
    /// The alpha value for the ELU activation function.
    /// </summary>
    [Description("The alpha value for the ELU activation function")]
    public double Alpha { get; set; } = 1D;

    /// <summary>
    /// If set to true, will do this operation in-place.
    /// </summary>
    [Description("If set to true, will do this operation in-place")]
    public bool Inplace { get; set; } = false;

    /// <summary>
    /// Creates an exponential linear unit (ELU) module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(ELU(Alpha, Inplace));
    }

    /// <summary>
    /// Creates an exponential linear unit (ELU) module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => ELU(Alpha, Inplace));
    }
}
