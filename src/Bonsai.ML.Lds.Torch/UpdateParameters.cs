using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

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
    /// The Kalman filter model.
    /// </summary>
    [XmlIgnore]
    [Description("The Kalman filter model.")]
    public KalmanFilter Model { get; set; }

    /// <summary>
    /// Updates the parameters of a Kalman filter model using elements from the input sequence.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<KalmanFilterParameters> Process(IObservable<KalmanFilterParameters> source)
    {
        return source.Do(Model.UpdateParameters);
    }
}