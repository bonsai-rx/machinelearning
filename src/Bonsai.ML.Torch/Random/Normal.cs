using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Random;

/// <summary>
/// Creates a tensor filled with random floats sampled from a normal distribution with mean 0 and variance 1.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a tensor filled with random floats sampled from a normal distribution.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Normal
{
    /// <summary>
    /// The size of the tensor.
    /// </summary>
    [Description("The size of the tensor.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] Size { get; set; }

    /// <summary>
    /// The data type of the tensor elements.
    /// </summary>
    [Description("The data type of the tensor elements.")]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// The device on which to create the tensor.
    /// </summary>
    [Description("The device on which to create the tensor.")]
    [XmlIgnore]
    public Device Device { get; set; } = null;

    /// <summary>
    /// The random number generator to use.
    /// </summary>
    [XmlIgnore]
    [Description("The random number generator to use.")]
    public Generator Generator { get; set; } = null;

    /// <summary>
    /// The mean of the normal distribution.
    /// </summary>
    [Description("The mean of the normal distribution.")]
    public double Mean { get; set; } = 0;

    /// <summary>
    /// The variance of the normal distribution.
    /// </summary>
    [Description("The variance of the normal distribution.")]
    public double Variance { get; set; } = 1;

    /// <summary>
    /// Creates a tensor filled with random values sampled from a normal distribution with mean 0 and variance 1.
    /// </summary>
    public IObservable<Tensor> Process()
    {
        return Observable.Return(randn(Size, dtype: Type, device: Device, generator: Generator) * Variance + Mean);
    }

    /// <summary>
    /// Generates an observable sequence of tensors filled with random values and uses the input generator.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Generator> source)
    {
        return source.Select(value =>
        {
            Generator = value;
            return randn(Size, dtype: Type, device: Device, generator: Generator) * Variance + Mean;
        });
    }

    /// <summary>
    /// Generates an observable sequence of tensors filled with random values for each element of the input sequence.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(value => randn_like(value, dtype: Type, device: Device) * Variance + Mean);
    }

    /// <summary>
    /// Generates an observable sequence of tensors filled with random values for each element of the input sequence.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process<T>(IObservable<T> source)
    {
        return source.Select(value => randn(Size, dtype: Type, device: Device, generator: Generator) * Variance + Mean);
    }
}
