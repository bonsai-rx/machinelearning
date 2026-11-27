using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Applies a Kalman smoother to the input filtered result sequence.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Applies a Kalman smoother to the input filtered result sequence.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Smooth
{
    /// <summary>
    /// Gets or sets the Kalman filter model.
    /// </summary>
    [Description("The Kalman filter model.")]
    [XmlIgnore]
    public KalmanFilter Model { get; set; }

    /// <summary>
    /// Processes an observable sequence of filtered results, applying the Kalman smoother to each result.
    /// </summary>
    /// <param name="source">The sequence of filtered states, in time order.</param>
    /// <returns>A sequence of smoothed states.</returns>
    public IObservable<LinearDynamicalSystemState> Process(IObservable<FilteredState> source)
    {
        return source.Select(Model.Smooth);
    }

    /// <summary>
    /// Processes an observable sequence of tuples containing the components of a filtered result (predictedMean, predictedCovariance, updatedMean, updatedCovariance), applying the Kalman smoother to each result.
    /// </summary>
    /// <param name="source">The sequence of predicted mean, predicted covariance, updated mean and updated covariance tuples, in time order.</param>
    /// <returns>A sequence of smoothed states.</returns>
    public IObservable<LinearDynamicalSystemState> Process(IObservable<Tuple<Tensor, Tensor, Tensor, Tensor>> source)
    {
        return source.Select((input) =>
        {
            var filteredState = new FilteredState(
                predictedState: new LinearDynamicalSystemState(input.Item1, input.Item2),
                updatedState: new LinearDynamicalSystemState(input.Item3, input.Item4)
            );
            return Model.Smooth(filteredState);
        });
    }
}
