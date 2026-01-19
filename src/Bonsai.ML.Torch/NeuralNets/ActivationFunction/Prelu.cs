using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.ActivationFunction;

/// <summary>
/// Represents an operator that creates a parametric rectified linear unit (PReLU) activation function.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html"/> for more information.
/// </remarks>
[Description("Creates a parametric rectified linear unit (PReLU) activation function.")]
[DisplayName("PReLU")]
public class Prelu
{
    /// <summary>
    /// The number of parameters to learn.
    /// </summary>
    [Description("The number of parameters to learn.")]
    public long NumParameters { get; set; }

    /// <summary>
    /// The initial value for the learnable parameters.
    /// </summary>
    [Description("The initial value for the learnable parameters.")]
    public double InitialValue { get; set; } = 0.25D;

    /// <summary>
    /// The desired device of returned tensor.
    /// </summary>
    [XmlIgnore]
    [Description("The desired device of returned tensor.")]
    public Device Device { get; set; } = null;

    /// <summary>
    /// The desired data type of returned tensor.
    /// </summary>
    [Description("The desired data type of returned tensor.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Creates a parametric rectified linear unit (PReLU) activation function.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.PReLU> Process()
    {
        return Observable.Return(PReLU(NumParameters, InitialValue, Device, Type));
    }

    /// <summary>
    /// Creates a parametric rectified linear unit (PReLU) activation function.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.PReLU> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => PReLU(NumParameters, InitialValue, Device, Type));
    }
}
