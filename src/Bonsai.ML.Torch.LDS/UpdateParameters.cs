using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Xml.Serialization;
using Bonsai;
using Bonsai.ML.Torch;
using Bonsai.ML.Torch.NeuralNets;
using TorchSharp;
using TorchSharp.Modules;

namespace Bonsai.ML.Torch.LDS;

[Combinator]
[ResetCombinator]
[Description("Learn the parameters kalman filter using the batch EM update algorithm.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class UpdateParameters
{
    [TypeConverter(typeof(KalmanFilterNameConverter))]
    public string ModelName { get; set; } = "KalmanFilter";

    public IObservable<KalmanFilterParameters> Process(IObservable<KalmanFilterParameters> source)
    {
        return source.Do((input) =>
        {
            var kalmanFilter = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            using (KalmanFilterModelManager.Write(kalmanFilter))
            {
                kalmanFilter.UpdateParameters(input);
            }
        });
    }
}