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
[Description("Creates a tensor filled with random floats.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class RandomFloats
{
    /// <summary>
    /// The size of the tensor.
    /// </summary>
    [Description("The size of the tensor.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] Size { get; set; } = new long[0];

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
    public torch.Generator Generator { get; set; } = null;

    /// <summary>
    /// Creates a tensor filled with random values sampled from a normal distribution with mean 0 and variance 1.
    /// </summary>
    public IObservable<Tensor> Process()
    {
        return Observable.Return(randn(Size, dtype: Type, device: Device, generator: Generator));
    }

    /// <summary>
    /// Generates an observable sequence of tensors filled with random values and uses the input generator.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<torch.Generator> source)
    {
        return source.Select(value =>
        {
            Generator = value;
            return randn(Size, dtype: Type, device: Device, generator: Generator);
        });
    }

    /// <summary>
    /// Generates an observable sequence of tensors filled with random values for each element of the input sequence.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(value => randn_like(value, dtype: Type, device: Device));
    }

    /// <summary>
    /// Generates an observable sequence of tensors filled with random values for each element of the input sequence.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process<T>(IObservable<T> source)
    {
        return source.Select(value => randn(Size, dtype: Type, device: Device, generator: Generator));
    }
}
