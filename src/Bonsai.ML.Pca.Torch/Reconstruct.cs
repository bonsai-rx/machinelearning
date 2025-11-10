using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using Bonsai;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Reconstructs the input data using a PCA model.
/// </summary>
[Combinator]
[Description("Reconstructs the input data using a PCA model.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Reconstruct
{
    /// <summary>
    /// The PCA model used to reconstruct the input data.
    /// </summary>
    [Description("The PCA model used to reconstruct the input data.")]
    [XmlIgnore]
    public IPcaBaseModel? Model { get; set; }

    /// <summary>
    /// Reconstructs the input data using the specified PCA model.
    /// </summary>
    /// <param name="model"></param>
    /// <param name="data"></param>
    /// <returns></returns>
    private static Tensor ReconstructData(IPcaBaseModel model, Tensor data)
    {
        return model.Reconstruct(data);
    }

    /// <summary>
    /// Reconstructs the input data using the specified PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("The PCA model has not been specified.");
        }
        return source.Select(value =>
        {
            return ReconstructData(Model, value);
        });
    }

    /// <summary>
    /// Reconstructs the input data using the specified PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Pca, Tensor>> source)
    {
        return source.Select(value =>
        {
            return ReconstructData(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Reconstructs the input data using the specified PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Pca>> source)
    {
        return source.Select(value =>
        {
            return ReconstructData(value.Item2, value.Item1);
        });
    }

    /// <summary>
    /// Reconstructs the input data using the specified PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<ProbabilisticPca, Tensor>> source)
    {
        return source.Select(value =>
        {
            return ReconstructData(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Reconstructs the input data using the specified PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, ProbabilisticPca>> source)
    {
        return source.Select(value =>
        {
            return ReconstructData(value.Item2, value.Item1);
        });
    }

    /// <summary>
    /// Reconstructs the input data using the specified PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<OnlineProbabilisticPca, Tensor>> source)
    {
        return source.Select(value =>
        {
            return ReconstructData(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Reconstructs the input data using the specified PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, OnlineProbabilisticPca>> source)
    {
        return source.Select(value =>
        {
            return ReconstructData(value.Item2, value.Item1);
        });
    }
}
