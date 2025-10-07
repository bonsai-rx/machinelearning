using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Orthogonalizes the state and covariance estimates from a Kalman filter or smoother.
/// </summary>
[Combinator]
[Description("Orthogonalizes the state and covariance estimates from a Kalman filter or smoother.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Orthogonalize
{
    /// <summary>
    /// The name of the Kalman filter model to be used.
    /// </summary>
    [TypeConverter(typeof(KalmanFilterNameConverter))]
    [Description("The name of the Kalman filter model to be used.")]
    public string ModelName { get; set; } = "KalmanFilter";

    /// <summary>
    /// Processes an observable sequence of smoothed results, orthogonalizing the mean and covariance estimates.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<OrthogonalizedState> Process(IObservable<SmoothedState> source)
    {
        return source.Select(input =>
        {
            var kalmanFilter = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            var smoothedMean = input.SmoothedMean;
            var smoothedCovariance = input.SmoothedCovariance;
            return kalmanFilter.OrthogonalizeMeanAndCovariance(smoothedMean, smoothedCovariance);
        });
    }

    /// <summary>
    /// Processes an observable sequence of filtered results, orthogonalizing the mean and covariance estimates.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<OrthogonalizedState> Process(IObservable<FilteredState> source)
    {
        return source.Select(input =>
        {
            var kalmanFilter = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            var filteredMean = input.UpdatedMean;
            var filteredCovariance = input.UpdatedCovariance;
            return kalmanFilter.OrthogonalizeMeanAndCovariance(filteredMean, filteredCovariance);
        });
    }

    /// <summary>
    /// Processes an observable sequence of filtered results, orthogonalizing the mean and covariance estimates.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<OrthogonalizedState> Process(IObservable<LdsState> source)
    {
        return source.Select(input =>
        {
            var kalmanFilter = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            var mean = input.Mean;
            var covariance = input.Covariance;
            return kalmanFilter.OrthogonalizeMeanAndCovariance(mean, covariance);
        });
    }

    /// <summary>
    /// Processes an observable sequence of LDS states, orthogonalizing the mean and covariance estimates.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<OrthogonalizedState> Process(IObservable<ILdsState> source)
    {
        return source.Select(input =>
        {
            var kalmanFilter = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            var mean = input.Mean;
            var covariance = input.Covariance;
            return kalmanFilter.OrthogonalizeMeanAndCovariance(mean, covariance);
        });
    }

    /// <summary>
    /// Processes an observable sequence of mean and covariance tuples, orthogonalizing the mean and covariance estimates.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<OrthogonalizedState> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(input =>
        {
            var kalmanFilter = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            var mean = input.Item1;
            var covariance = input.Item2;
            return kalmanFilter.OrthogonalizeMeanAndCovariance(mean, covariance);
        });
    }
}