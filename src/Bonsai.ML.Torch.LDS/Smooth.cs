using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Applies a Kalman smoother to the input filtered result sequence.
/// </summary>
[Combinator]
[Description("Applies a Kalman smoother to the input filtered result sequence.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Smooth
{
    /// <summary>
    /// The name of the Kalman filter model to be used.
    /// </summary>
    [TypeConverter(typeof(KalmanFilterNameConverter))]
    [Description("The name of the Kalman filter model to be used.")]
    public string ModelName { get; set; } = "KalmanFilter";

    /// <summary>
    /// Processes an observable sequence of filtered results, applying the Kalman smoother to each result.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<SmoothedState> Process(IObservable<FilteredState> source)
    {
        return source.Select((input) =>
        {
            var kalmanFilter = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            return kalmanFilter.Smooth(input);
        });
    }

    /// <summary>
    /// Processes an observable sequence of tuples containing the components of a filtered result (predictedMean, predictedCovariance, updatedMean, updatedCovariance), applying the Kalman smoother to each result.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<SmoothedState> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select((input) =>
        {
            var kalmanFilter = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            var filteredState = new FilteredState(input.Item1, input.Item2, input.Item3, input.Item4);
            return kalmanFilter.Smooth(filteredState);
        });
    }
}