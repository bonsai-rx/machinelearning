using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates a Beta probability distribution parameterized by two concentration parameters (alpha, beta).
/// </summary>
[Combinator]
[Description("Creates a Beta distribution with concentration parameters (alpha, beta).")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class Beta : IScalarTypeProvider
{
    /// <summary>
    /// Concentration parameter alpha (> 0). Can be a scalar or tensor; the shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Concentration alpha (> 0). Can be a scalar or tensor; shape sets the batch/event shape of the distribution.")]
    public Tensor Concentration1 { get; set; } = null;

    /// <summary>
    /// The values of concentration 1 in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Concentration1))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string Concentration1Xml
    {
        get => TensorConverter.ConvertToString(Concentration1, Type);
        set => Concentration1 = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Concentration parameter beta (> 0). Can be a scalar or tensor; the shape determines the batch/event shape.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("Concentration beta (> 0). Can be a scalar or tensor; shape sets the batch/event shape of the distribution.")]
    public Tensor Concentration0 { get; set; } = null;

    /// <summary>
    /// The values of concentration 0 in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Concentration0))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string Concentration0Xml
    {
        get => TensorConverter.ConvertToString(Concentration0, Type);
        set => Concentration0 = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Gets or sets the data type of the tensor elements.
    /// </summary>
    [Description("The data type of the tensor elements.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Creates a Beta distribution.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Beta> Process()
    {
        return Observable.Return(distributions.Beta(Concentration1, Concentration0));
    }

    /// <summary>
    /// Creates a Beta distribution using the incoming RNG generator.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Beta> Process(IObservable<Generator> source)
    {
        return source.Select(generator => distributions.Beta(Concentration1, Concentration0, generator: generator));
    }

    /// <summary>
    /// For each element of the source stream, emits a Beta distribution.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Beta> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Beta(Concentration1, Concentration0));
    }
}