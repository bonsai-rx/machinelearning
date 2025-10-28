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
    public IObservable<LinearDynamicalSystemState> Process(IObservable<ILinearDynamicalSystemState> source)
    {
        return source.Select(input =>
        {
            return Model.OrthogonalizeMeanAndCovariance(input.Mean, input.Covariance);
        });
    }

    /// <summary>
    /// Processes an observable sequence of filtered results, orthogonalizing the mean and covariance estimates.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LinearDynamicalSystemState> Process(IObservable<FilteredState> source)
    {
        return source.Select(input =>
        {
            return Model.OrthogonalizeMeanAndCovariance(input.Mean, input.Covariance);
        });
    }

    /// <summary>
    /// Processes an observable sequence of filtered results, orthogonalizing the mean and covariance estimates.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LinearDynamicalSystemState> Process(IObservable<LinearDynamicalSystemState> source)
    {
        return source.Select(input =>
        {
            return Model.OrthogonalizeMeanAndCovariance(input.Mean, input.Covariance);
        });
    }

    /// <summary>
    /// Processes an observable sequence of mean and covariance tuples, orthogonalizing the mean and covariance estimates.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LinearDynamicalSystemState> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(input =>
        {
            var mean = input.Item1;
            var covariance = input.Item2;
            return Model.OrthogonalizeMeanAndCovariance(mean, covariance);
        });
    }
}