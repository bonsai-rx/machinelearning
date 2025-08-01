using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class PrintTensor
{
    public TensorStringStyle StringStyle { get; set; }
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Do(value => Console.WriteLine(value.ToString(StringStyle)));
    }
}
