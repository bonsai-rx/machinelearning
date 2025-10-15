using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Orthogonalizes the state and covariance estimates from a Kalman filter or smoother.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Orthogonalizes the state and covariance estimates from a Kalman filter or smoother.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Orthogonalize
{
    /// <summary>
    /// The Kalman filter model.
    /// </summary>
    [Description("The Kalman filter model.")]
    [XmlIgnore]
    public KalmanFilter Model { get; set; }

    /// <summary>
    /// Processes an observable sequence of smoothed results, orthogonalizing the mean and covariance estimates.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<OrthogonalizedState> Process(IObservable<SmoothedState> source)
    {
        return source.Select(input =>
        {
            var smoothedMean = input.SmoothedMean;
            var smoothedCovariance = input.SmoothedCovariance;
            return Model.OrthogonalizeMeanAndCovariance(smoothedMean, smoothedCovariance);
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
            var filteredMean = input.UpdatedMean;
            var filteredCovariance = input.UpdatedCovariance;
            return Model.OrthogonalizeMeanAndCovariance(filteredMean, filteredCovariance);
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
            var mean = input.Mean;
            var covariance = input.Covariance;
            return Model.OrthogonalizeMeanAndCovariance(mean, covariance);
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
            var mean = input.Mean;
            var covariance = input.Covariance;
            return Model.OrthogonalizeMeanAndCovariance(mean, covariance);
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
            var mean = input.Item1;
            var covariance = input.Item2;
            return Model.OrthogonalizeMeanAndCovariance(mean, covariance);
        });
    }
}