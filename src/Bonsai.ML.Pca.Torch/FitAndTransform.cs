using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using Bonsai;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Fits the PCA model to the input data and transforms it.
/// </summary>
[Combinator]
[Description("Fits the PCA model to the input data and transforms it.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class FitAndTransform
{
    /// <summary>
    /// The PCA model used to fit and transform the input data.
    /// </summary>
    [Description("The PCA model used to fit and transform the input data.")]
    [XmlIgnore]
    public IPcaBaseModel? Model { get; set; }

    private static void FitModelAndTransformData(IPcaBaseModel model, Tensor data)
    {
        model.FitAndTransform(data);
    }


    /// <summary>
    /// Fits the PCA model to the input data and transforms it.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("The PCA model has not been specified.");
        }
        return source.Do(value =>
        {
            FitModelAndTransformData(Model, value);
        });
    }

    /// <summary>
    /// Fits the PCA model to the input data and transforms it.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<Pca, Tensor>> Process(IObservable<Tuple<Pca, Tensor>> source)
    {
        return source.Do((value) =>
        {
            FitModelAndTransformData(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Fits the PCA model to the input data and transforms it.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<Tensor, Pca>> Process(IObservable<Tuple<Tensor, Pca>> source)
    {
        return source.Do((value) =>
        {
            FitModelAndTransformData(value.Item2, value.Item1);
        });
    }

    /// <summary>
    /// Fits the PCA model to the input data and transforms it.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<ProbabilisticPca, Tensor>> Process(IObservable<Tuple<ProbabilisticPca, Tensor>> source)
    {
        return source.Do((value) =>
        {
            FitModelAndTransformData(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Fits the PCA model to the input data and transforms it.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<Tensor, ProbabilisticPca>> Process(IObservable<Tuple<Tensor, ProbabilisticPca>> source)
    {
        return source.Do((value) =>
        {
            FitModelAndTransformData(value.Item2, value.Item1);
        });
    }

    /// <summary>
    /// Fits the PCA model to the input data and transforms it.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<OnlineProbabilisticPca, Tensor>> Process(IObservable<Tuple<OnlineProbabilisticPca, Tensor>> source)
    {
        return source.Do((value) =>
        {
            FitModelAndTransformData(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Fits the PCA model to the input data and transforms it.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<Tensor, OnlineProbabilisticPca>> Process(IObservable<Tuple<Tensor, OnlineProbabilisticPca>> source)
    {
        return source.Do((value) =>
        {
            FitModelAndTransformData(value.Item2, value.Item1);
        });
    }
}
