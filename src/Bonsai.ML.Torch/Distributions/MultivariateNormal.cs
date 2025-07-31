using static TorchSharp.torch;
using TorchSharp;
using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.Distributions;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Source)]
public class MultivariateNormal : IScalarTypeProvider
{
    [XmlIgnore]
    [Browsable(false)]
    public ScalarType Type => ScalarType.Float32;

    [TypeConverter(typeof(TensorConverter))]
    public Tensor Mean { get; set; }

    [TypeConverter(typeof(TensorConverter))]
    public Tensor Covariance { get; set; }

    [XmlIgnore]
    public torch.Generator Generator { get; set; } = null;

    public IObservable<TorchSharp.Modules.MultivariateNormal> Process()
    {
        return Observable.Return(distributions.MultivariateNormal(Mean, Covariance, generator: Generator));
    }

    public IObservable<TorchSharp.Modules.MultivariateNormal> Process(IObservable<torch.Generator> source)
    {
        return source.Select((generator) =>
        {
            Generator = generator;
            return distributions.MultivariateNormal(Mean, Covariance, generator: Generator);
        });
    }

    public IObservable<TorchSharp.Modules.MultivariateNormal> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.MultivariateNormal(Mean, Covariance, generator: Generator));
    }
}