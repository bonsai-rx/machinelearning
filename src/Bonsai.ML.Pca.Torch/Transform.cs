using System;
using System.ComponentModel;
using System.Reactive.Linq;
using Bonsai;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

[Combinator]
[Description]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Transform
{
    private Tensor TransformData(IPcaBaseModel model, Tensor data)
    {
        return model.Transform(data);
    }

    public IObservable<Tensor> Process(IObservable<Tuple<Pca, Tensor>> source)
    {
        return source.Select(value =>
        {
            return TransformData(value.Item1, value.Item2);
        });
    }

    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, Pca>> source)
    {
        return source.Select(value =>
        {
            return TransformData(value.Item2, value.Item1);
        });
    }

    public IObservable<Tensor> Process(IObservable<Tuple<ProbabilisticPca, Tensor>> source)
    {
        return source.Select(value =>
        {
            return TransformData(value.Item1, value.Item2);
        });
    }

    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, ProbabilisticPca>> source)
    {
        return source.Select(value =>
        {
            return TransformData(value.Item2, value.Item1);
        });
    }

    public IObservable<Tensor> Process(IObservable<Tuple<OnlineProbabilisticPca, Tensor>> source)
    {
        return source.Select(value =>
        {
            return TransformData(value.Item1, value.Item2);
        });
    }

    public IObservable<Tensor> Process(IObservable<Tuple<Tensor, OnlineProbabilisticPca>> source)
    {
        return source.Select(value =>
        {
            return TransformData(value.Item2, value.Item1);
        });
    }
}
