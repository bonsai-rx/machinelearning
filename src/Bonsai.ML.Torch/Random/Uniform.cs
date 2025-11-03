using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Random;

/// <summary>
/// Creates a tensor filled with random numbers sampled from a uniform distribution over the interval [MinSize, MaxSize).
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a tensor filled with random numbers from a uniform distribution over the interval [MinSize, MaxSize).")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Uniform
{
    /// <summary>
    /// The size of the tensor.
    /// </summary>
    [Description("The size of the tensor.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] Size { get; set; } = [];

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
    /// The minimum value of the random numbers inclusive.
    /// </summary>
    [Description("The minimum value of the random numbers.")]
    public double MinValue { get; set; } = 0;

    /// <summary>
    /// The maximum value of the random numbers exclusive.
    /// </summary>
    [Description("The maximum value of the random numbers.")]
    public double MaxValue { get; set; } = 1;


    /// <summary>
    /// Creates a tensor filled with random values sampled from a uniform distribution over the interval [MinValue, MaxValue).
    /// </summary>
    public IObservable<Tensor> Process()
    {
        return Observable.Return(rand(Size, dtype: Type, device: Device, generator: Generator) * (MaxValue - MinValue) + MinValue);
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
            return rand(Size, dtype: Type, device: Device, generator: Generator) * (MaxValue - MinValue) + MinValue;
        });
    }

    /// <summary>
    /// Generates an observable sequence of tensors filled with random values for each element of the input sequence.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(value => rand_like(value, dtype: Type, device: Device) * (MaxValue - MinValue) + MinValue);
    }

    /// <summary>
    /// Generates an observable sequence of tensors filled with random values for each element of the input sequence.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process<T>(IObservable<T> source)
    {
        return source.Select(value => rand(Size, dtype: Type, device: Device, generator: Generator) * (MaxValue - MinValue) + MinValue);
    }

}
