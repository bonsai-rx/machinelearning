using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using Bonsai;
using Bonsai.ML.Torch;
using Bonsai.ML.Torch.NeuralNets;
using TorchSharp;
using TorchSharp.Modules;

namespace Bonsai.ML.Torch.LDS;

[Combinator]
[ResetCombinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Smooth
{
    [TypeConverter(typeof(KalmanFilterNameConverter))]
    public string ModelName { get; set; } = "KalmanFilter";

    public IObservable<SmoothedResult> Process(IObservable<FilteredResult> source)
    {
        return source.Select((input) =>
        {
            var kalmanFilter = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            using (KalmanFilterModelManager.Read(kalmanFilter))
            {
                return kalmanFilter.Smooth(input);
            }
        });
    }
}