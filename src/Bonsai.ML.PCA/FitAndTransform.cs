using System;
using System.ComponentModel;
using System.Reactive.Linq;
using Bonsai;
using static TorchSharp.torch;

namespace Bonsai.ML.PCA
{
    [Combinator]
    [Description]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class FitAndTransform
    {
        private void FitModelAndTransformData(IPCABaseModel model, Tensor data)
        {
            model.FitAndTransform(data);
        }

        public IObservable<Tuple<PCA, Tensor>> Process(IObservable<Tuple<PCA, Tensor>> source)
        {
            return source.Do((value) =>
            {
                FitModelAndTransformData(value.Item1, value.Item2);
            });
        }

        public IObservable<Tuple<Tensor, PCA>> Process(IObservable<Tuple<Tensor, PCA>> source)
        {
            return source.Do((value) =>
            {
                FitModelAndTransformData(value.Item2, value.Item1);
            });
        }

        public IObservable<Tuple<PPCA, Tensor>> Process(IObservable<Tuple<PPCA, Tensor>> source)
        {
            return source.Do((value) =>
            {
                FitModelAndTransformData(value.Item1, value.Item2);
            });
        }

        public IObservable<Tuple<Tensor, PPCA>> Process(IObservable<Tuple<Tensor, PPCA>> source)
        {
            return source.Do((value) =>
            {
                FitModelAndTransformData(value.Item2, value.Item1);
            });
        }
        
        public IObservable<Tuple<OnlinePPCA, Tensor>> Process(IObservable<Tuple<OnlinePPCA, Tensor>> source)
        {
            return source.Do((value) =>
            {
                FitModelAndTransformData(value.Item1, value.Item2);
            });
        }
        
        public IObservable<Tuple<Tensor, OnlinePPCA>> Process(IObservable<Tuple<Tensor, OnlinePPCA>> source)
        {
            return source.Do((value) =>
            {
                FitModelAndTransformData(value.Item2, value.Item1);
            });
        }
    }
}
