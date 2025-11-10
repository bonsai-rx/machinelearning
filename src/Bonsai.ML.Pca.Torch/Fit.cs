using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using Bonsai;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Fits a PCA model to the input data.
/// </summary>
[Combinator]
[Description("Fits a PCA model to the input data.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class Fit
{
    /// <summary>
    /// The PCA model used to fit the input data.
    /// </summary>
    [Description("The PCA model used to fit the input data.")]
    [XmlIgnore]
    public IPcaBaseModel? Model { get; set; }

    private void FitModel(IPcaBaseModel model, Tensor data)
    {
        model.Fit(data);
    }

    /// <summary>
    /// Fits the PCA model to the input data.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("The PCA model has not been specified.");
        }
        return source.Do(value =>
        {
            FitModel(Model, value);
        });
    }

    /// <summary>
    /// Fits the PCA model to the input data.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<Pca, Tensor>> Process(IObservable<Tuple<Pca, Tensor>> source)
    {
        return source.Do((value) =>
        {
            FitModel(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Fits the PCA model to the input data.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<Tensor, Pca>> Process(IObservable<Tuple<Tensor, Pca>> source)
    {
        return source.Do((value) =>
        {
            FitModel(value.Item2, value.Item1);
        });
    }

    /// <summary>
    /// Fits the PCA model to the input data.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<ProbabilisticPca, Tensor>> Process(IObservable<Tuple<ProbabilisticPca, Tensor>> source)
    {
        return source.Do((value) =>
        {
            FitModel(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Fits the PCA model to the input data.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<Tensor, ProbabilisticPca>> Process(IObservable<Tuple<Tensor, ProbabilisticPca>> source)
    {
        return source.Do((value) =>
        {
            FitModel(value.Item2, value.Item1);
        });
    }

    /// <summary>
    /// Fits the PCA model to the input data.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<OnlineProbabilisticPca, Tensor>> Process(IObservable<Tuple<OnlineProbabilisticPca, Tensor>> source)
    {
        return source.Do((value) =>
        {
            FitModel(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Fits the PCA model to the input data.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<Tensor, OnlineProbabilisticPca>> Process(IObservable<Tuple<Tensor, OnlineProbabilisticPca>> source)
    {
        return source.Do((value) =>
        {
            FitModel(value.Item2, value.Item1);
        });
    }
}
