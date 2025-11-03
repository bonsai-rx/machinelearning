using System;
using System.ComponentModel;
using System.Reactive.Linq;
using Bonsai;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

[Combinator]
[Description]
[WorkflowElementCategory(ElementCategory.Transform)]
public class FitAndTransform
{
    private void FitModelAndTransformData(IPcaBaseModel model, Tensor data)
    {
        model.FitAndTransform(data);
    }

    public IObservable<Tuple<Pca, Tensor>> Process(IObservable<Tuple<Pca, Tensor>> source)
    {
        return source.Do((value) =>
        {
            FitModelAndTransformData(value.Item1, value.Item2);
        });
    }

    public IObservable<Tuple<Tensor, Pca>> Process(IObservable<Tuple<Tensor, Pca>> source)
    {
        return source.Do((value) =>
        {
            FitModelAndTransformData(value.Item2, value.Item1);
        });
    }

    public IObservable<Tuple<ProbabilisticPca, Tensor>> Process(IObservable<Tuple<ProbabilisticPca, Tensor>> source)
    {
        return source.Do((value) =>
        {
            FitModelAndTransformData(value.Item1, value.Item2);
        });
    }

    public IObservable<Tuple<Tensor, ProbabilisticPca>> Process(IObservable<Tuple<Tensor, ProbabilisticPca>> source)
    {
        return source.Do((value) =>
        {
            FitModelAndTransformData(value.Item2, value.Item1);
        });
    }

    public IObservable<Tuple<OnlineProbabilisticPca, Tensor>> Process(IObservable<Tuple<OnlineProbabilisticPca, Tensor>> source)
    {
        return source.Do((value) =>
        {
            FitModelAndTransformData(value.Item1, value.Item2);
        });
    }

    public IObservable<Tuple<Tensor, OnlineProbabilisticPca>> Process(IObservable<Tuple<Tensor, OnlineProbabilisticPca>> source)
    {
        return source.Do((value) =>
        {
            FitModelAndTransformData(value.Item2, value.Item1);
        });
    }
}
