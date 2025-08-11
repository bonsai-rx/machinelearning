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
    public class Transform
    {
        private Tensor TransformData(IPCABaseModel model, Tensor data)
        {
            return model.Transform(data);
        }

        public IObservable<Tensor> Process(IObservable<Tuple<PCA, Tensor>> source)
        {
            return source.Select(value =>
            {
                return TransformData(value.Item1, value.Item2);
            });
        }

        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, PCA>> source)
        {
            return source.Select(value =>
            {
                return TransformData(value.Item2, value.Item1);
            });
        }

        public IObservable<Tensor> Process(IObservable<Tuple<PPCA, Tensor>> source)
        {
            return source.Select(value =>
            {
                return TransformData(value.Item1, value.Item2);
            });
        }

        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, PPCA>> source)
        {
            return source.Select(value =>
            {
                return TransformData(value.Item2, value.Item1);
            });
        }

        public IObservable<Tensor> Process(IObservable<Tuple<OnlinePPCA, Tensor>> source)
        {
            return source.Select(value =>
            {
                return TransformData(value.Item1, value.Item2);
            });
        }

        public IObservable<Tensor> Process(IObservable<Tuple<Tensor, OnlinePPCA>> source)
        {
            return source.Select(value =>
            {
                return TransformData(value.Item2, value.Item1);
            });
        }
    }
}
