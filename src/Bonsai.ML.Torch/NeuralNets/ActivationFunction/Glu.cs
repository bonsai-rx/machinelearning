using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a gated linear unit (GLU) module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.GLU.html"/> for more information.
/// </remarks>
[Description("Creates a gated linear unit (GLU) module.")]
[DisplayName("GLU")]
public class Glu
{
    /// <summary>
    /// The dimension on which to split the input tensor.
    /// </summary>
    [Description("The dimension on which to split the input tensor.")]
    public long Dim { get; set; } = -1;

    /// <summary>
    /// Creates a gated linear unit (GLU) module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.GLU> Process()
    {
        return Observable.Return(GLU(Dim));
    }

    /// <summary>
    /// Creates a gated linear unit (GLU) module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.GLU> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => GLU(Dim));
    }
}
