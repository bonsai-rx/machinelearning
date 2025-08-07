using System;
using System.ComponentModel;
using System.Reactive.Linq;
using Bonsai;
using static TorchSharp.torch;

namespace Bonsai.ML.PCA
{
    [Combinator]
    [Description]
    [WorkflowElementCategory(ElementCategory.Sink)]
    public class Fit
    {
        private void FitModel(IPCABaseModel model, Tensor data)
        {
            model.Fit(data);
        }

        public IObservable<Tuple<PCA, Tensor>> Process(IObservable<Tuple<PCA, Tensor>> source)
        {
            return source.Do((value) =>
            {
                FitModel(value.Item1, value.Item2);
            });
        }
        public IObservable<Tuple<Tensor, PCA>> Process(IObservable<Tuple<Tensor, PCA>> source)
        {
            return source.Do((value) =>
            {
                FitModel(value.Item2, value.Item1);
            });
        }
    }
}
