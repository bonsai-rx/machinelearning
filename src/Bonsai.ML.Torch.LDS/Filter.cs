using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Applies a Kalman filter to the input tensor sequence.
/// </summary>
[Combinator]
[Description("Applies a Kalman filter to the input tensor sequence.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Filter
{
    /// <summary>
    /// The name of the Kalman filter model to be used.
    /// </summary>
    [TypeConverter(typeof(KalmanFilterNameConverter))]
    [Description("The name of the Kalman filter model to be used.")]
    public string ModelName { get; set; } = "KalmanFilter";

    /// <summary>
    /// Processes an observable sequence of input tensors, applying the Kalman filter to each tensor.
    /// </summary>
    public IObservable<FilteredResult> Process(IObservable<Tensor> source)
    {
        return source.Select((input) =>
        {
            var kalmanFilter = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            using (KalmanFilterModelManager.Read(kalmanFilter))
            {
                return kalmanFilter.Filter(input);
            }
        });
    }
}