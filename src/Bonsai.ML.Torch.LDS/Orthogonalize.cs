using TorchSharp;
using System;
using Bonsai;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.LDS;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Orthogonalize
{
    [TypeConverter(typeof(KalmanFilterNameConverter))]
    public string ModelName { get; set; } = "KalmanFilter";

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
}