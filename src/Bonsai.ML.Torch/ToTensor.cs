using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Xml.Serialization;
using OpenCV.Net;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch;

/// <summary>
/// Converts the input value into a tensor.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Converts the input value into a tensor.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class ToTensor
{
    /// <summary>
    /// The device on which to create the tensor.
    /// </summary>
    [Description("The device on which to create the tensor.")]
    [XmlIgnore]
    public Device Device { get; set; } = null;

    /// <summary>
    /// The data type of the tensor.
    /// </summary>
    [Description("The data type of the tensor.")]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Converts an int into a tensor.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<int> source)
    {
        return source.Select(value => as_tensor(value, dtype: Type, device: Device));
    }

    /// <summary>
    /// Converts a double into a tensor.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<double> source)
    {
        return source.Select(value => as_tensor(value, dtype: Type, device: Device));
    }

    /// <summary>
    /// Converts a byte into a tensor.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<byte> source)
    {
        return source.Select(value => as_tensor(value, dtype: Type, device: Device));
    }

    /// <summary>
    /// Converts a bool into a tensor.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<bool> source)
    {
        return source.Select(value => as_tensor(value, dtype: Type, device: Device));
    }

    /// <summary>
    /// Converts a float into a tensor.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<float> source)
    {
        return source.Select(value => as_tensor(value, dtype: Type, device: Device));
    }

    /// <summary>
    /// Converts a long into a tensor.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<long> source)
    {
        return source.Select(value => as_tensor(value, dtype: Type, device: Device));
    }

    /// <summary>
    /// Converts a short into a tensor.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<short> source)
    {
        return source.Select(value => as_tensor(value, dtype: Type, device: Device));
    }

    /// <summary>
    /// Converts an array into a tensor.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Array> source)
    {
        return source.Select(value => as_tensor(value, dtype: Type, device: Device));
    }

    /// <summary>
    /// Converts an IplImage into a tensor.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<IplImage> source)
    {
        return source.Select(value => OpenCVHelper.ToTensor(value, device: Device));
    }

    /// <summary>
    /// Converts a Mat into a tensor.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Mat> source)
    {
        return source.Select(value => OpenCVHelper.ToTensor(value, device: Device));
    }
}