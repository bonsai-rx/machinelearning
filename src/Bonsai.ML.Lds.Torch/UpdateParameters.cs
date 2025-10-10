using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Updates the parameters of a Kalman filter model instance using the provided Kalman filter parameters.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Updates the parameters of a Kalman filter model instance using the provided Kalman filter parameters.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class UpdateParameters
{
    /// <summary>
    /// The name of the Kalman filter model to be used.
    /// </summary>
    [TypeConverter(typeof(KalmanFilterNameConverter))]
    [Description("The name of the Kalman filter model to be used.")]
    public string ModelName { get; set; } = "KalmanFilter";

    /// <summary>
    /// Updates the parameters of a Kalman filter model using elements from the input sequence.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<KalmanFilterParameters> Process(IObservable<KalmanFilterParameters> source)
    {
        return source.Do((input) =>
        {
            var kalmanFilter = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            kalmanFilter.UpdateParameters(input);
        });
    }
}