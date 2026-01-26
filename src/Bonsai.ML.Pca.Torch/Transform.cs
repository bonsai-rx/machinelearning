using System;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Transforms the input data using a PCA model.
/// </summary>
public class Transform : IPcaModelProvider
{
    /// <inheritdoc/>
    public IPcaBaseModel? Model { get; set; } = null;

    private static Tensor TransformData(IPcaBaseModel model, Tensor data)
    {
        return model.Transform(data);
    }

    /// <summary>
    /// Transforms the input data.
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
            return TransformData(Model, value);
        });
    }

    /// <summary>
    /// Transforms the input data using a standard PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Pca, Tensor>> source)
    {
        return source.Select(value =>
        {
            return TransformData(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Transforms the input data using a standard PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Pca>> source)
    {
        return source.Select(value =>
        {
            return TransformData(value.Item2, value.Item1);
        });
    }

    /// <summary>
    /// Transforms the input data using a standard PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<ProbabilisticPca, Tensor>> source)
    {
        return source.Select(value =>
        {
            return TransformData(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Transforms the input data using a probabilistic PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, ProbabilisticPca>> source)
    {
        return source.Select(value =>
        {
            return TransformData(value.Item2, value.Item1);
        });
    }

    /// <summary>
    /// Transforms the input data using an online probabilistic PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<OnlineProbabilisticPca, Tensor>> source)
    {
        return source.Select(value =>
        {
            return TransformData(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Transforms the input data using an online probabilistic PCA model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, OnlineProbabilisticPca>> source)
    {
        return source.Select(value =>
        {
            return TransformData(value.Item2, value.Item1);
        });
    }

    /// <summary>
    /// Transforms the input data using an online PCA model based on the Generalized Hebbian Algorithm.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<OnlinePcaGha, Tensor>> source)
    {
        return source.Select(value =>
        {
            return TransformData(value.Item1, value.Item2);
        });
    }

    /// <summary>
    /// Transforms the input data using an online PCA model based on the Generalized Hebbian Algorithm.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, OnlinePcaGha>> source)
    {
        return source.Select(value =>
        {
            return TransformData(value.Item2, value.Item1);
        });
    }
}
