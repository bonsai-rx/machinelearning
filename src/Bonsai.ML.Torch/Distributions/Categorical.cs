using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Distributions;

/// <summary>
/// Creates a categorical (discrete) distribution over classes given event probabilities.
/// </summary>
[Combinator]
[Description("Creates a categorical (discrete) distribution over classes given event probabilities.")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class Categorical : IScalarTypeProvider
{
    /// <summary>
    /// The class probabilities. Values must be non-negative and typically sum to 1 per row. Can be a 1D vector or higher-rank tensor for batched distributions.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("The class probabilities. Values must be non-negative and typically sum to 1 per row. Can be a 1D vector or higher-rank tensor for batched distributions.")]
    public Tensor Probabilities { get; set; } = null;

    /// <summary>
    /// The values of probabilities in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Probabilities))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ProbabilitiesXml
    {
        get => TensorConverter.ConvertToString(Probabilities, Type);
        set => Probabilities = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Gets or sets the data type of the tensor elements.
    /// </summary>
    [Description("The data type of the tensor elements.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Creates a categorical distribution.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Categorical> Process()
    {
        return Observable.Return(distributions.Categorical(Probabilities));
    }

    /// <summary>
    /// Creates a categorical distribution for each incoming RNG generator.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Categorical> Process(IObservable<Generator> source)
    {
        return source.Select(generator => distributions.Categorical(Probabilities, generator: generator));
    }

    /// <summary>
    /// For each element of the source stream, emits a categorical distribution.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.Categorical> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => distributions.Categorical(Probabilities));
    }
}
