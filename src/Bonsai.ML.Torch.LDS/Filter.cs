using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using Bonsai;
using TorchSharp;

namespace Bonsai.ML.Torch.LDS;

[Combinator]
[ResetCombinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Filter
{
    [TypeConverter(typeof(KalmanFilterNameConverter))]
    public string ModelName { get; set; } = "KalmanFilter";

    public IObservable<FilteredResult> Process(IObservable<torch.Tensor> source)
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