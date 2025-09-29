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
    /// Processes an observable sequence of smoothed results, orthogonalizing the state and covariance estimates.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<OrthogonalizedResult> Process(IObservable<SmoothedResult> source)
    {
        return source.Select(input =>
        {
            var kalmanFilter = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            var smoothedState = input.SmoothedState;
            var smoothedCovariance = input.SmoothedCovariance;
            using (KalmanFilterModelManager.Read(kalmanFilter))
            {
                return kalmanFilter.OrthogonalizeStateAndCovariance(smoothedState, smoothedCovariance);
            }
        });
    }

    /// <summary>
    /// Processes an observable sequence of filtered results, orthogonalizing the state and covariance estimates.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<OrthogonalizedResult> Process(IObservable<FilteredResult> source)
    {
        return source.Select(input =>
        {
            var kalmanFilter = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            var filteredState = input.UpdatedState;
            var filteredCovariance = input.UpdatedCovariance;
            using (KalmanFilterModelManager.Read(kalmanFilter))
            {
                return kalmanFilter.OrthogonalizeStateAndCovariance(filteredState, filteredCovariance);
            }
        });
    }

    /// <summary>
    /// Processes an observable sequence of state and covariance tuples, orthogonalizing the state and covariance estimates.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<OrthogonalizedResult> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(input =>
        {
            var kalmanFilter = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            var state = input.Item1;
            var covariance = input.Item2;
            using (KalmanFilterModelManager.Read(kalmanFilter))
            {
                return kalmanFilter.OrthogonalizeStateAndCovariance(state, covariance);
            }
        });
    }
}