using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Applies a Kalman filter to the input tensor sequence.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Applies a Kalman filter to the input tensor sequence.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Filter
{
    /// <summary>
    /// The Kalman filter model.
    /// </summary>
    [Description("The Kalman filter model.")]
    [XmlIgnore]
    public KalmanFilter Model { get; set; }

    /// <summary>
    /// Processes an observable sequence of input tensors, applying the Kalman filter to each tensor.
    /// </summary>
    public IObservable<FilteredState> Process(IObservable<Tensor> source)
    {
        return source.Select(Model.Filter);
    }
}