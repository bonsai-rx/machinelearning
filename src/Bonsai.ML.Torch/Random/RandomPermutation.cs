using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Random;

/// <summary>
/// Creates a 1D tensor of a given size with a random permutation of integers from 0 to size - 1.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a 1D tensor of a given size with a random permutation of integers from 0 to size - 1.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class RandomPermutation
{
    /// <summary>
    /// The size of the tensor.
    /// </summary>
    [Description("The size of the tensor.")]
    public long Size { get; set; } = 0;

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
    /// Creates a tensor of a given size with a random permutation of integers from 0 to size - 1.
    /// </summary>
    public IObservable<Tensor> Process()
    {
        return Observable.Return(randperm(Size, dtype: Type, device: Device, generator: Generator));
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
            return randperm(Size, dtype: Type, device: Device, generator: Generator);
        });
    }

    /// <summary>
    /// Generates an observable sequence of tensors filled with random values for each element of the input sequence.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process<T>(IObservable<T> source)
    {
        return source.Select(value => randperm(Size, dtype: Type, device: Device, generator: Generator));
    }
}
