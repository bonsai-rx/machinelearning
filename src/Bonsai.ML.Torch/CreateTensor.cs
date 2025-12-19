using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Linq.Expressions;
using System.Reactive.Linq;
using System.Reflection;
using System.Xml.Serialization;
using Bonsai.Expressions;
using static TorchSharp.torch;
using Bonsai.ML.Data;
using TorchSharp;

namespace Bonsai.ML.Torch;

/// <summary>
/// Creates a tensor from the specified values.
/// Uses Python-like syntax to specify the tensor values. 
/// For example, a 2x2 tensor can be created with the following values: "[[1, 2], [3, 4]]".
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a tensor from the specified values. Uses Python-like syntax to specify the tensor values. For example, a 2x2 tensor can be created with the following values: \"[[1, 2], [3, 4]]\".")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class CreateTensor : IScalarTypeProvider
{
    /// <summary>
    /// The data type of the tensor elements.
    /// </summary>
    [Description("The data type of the tensor elements.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// The values of the tensor elements. 
    /// Uses Python-like syntax to specify the tensor values.
    /// For example: "[[1, 2], [3, 4]]".
    /// </summary>
    [XmlIgnore]
    [Description("The values of the tensor elements. Uses Python-like syntax to specify the tensor values. For example: \"[[1, 2], [3, 4]]\".")]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Values
    {
        get => _values;
        set => _values = value;    
    }

    /// <summary>
    /// The values of the tensor elements in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Values))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ValuesXml
    {
        get => TensorConverter.ConvertToString(Values, Type);
        set => Values = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The device on which to create the tensor.
    /// </summary>
    [XmlIgnore]
    [Description("The device on which to create the tensor.")]
    public Device Device { get; set; } = null;

    private Tensor _values;

    /// <summary>
    /// Returns an observable sequence that creates a tensor from the specified values.
    /// </summary>
    public IObservable<Tensor> Process()
    {
        var device = Device ?? CPU;
        return Observable.Return(_values.to(device).clone());
    }

    /// <summary>
    /// Returns an observable sequence that creates a tensor from the specified values for each element in the input sequence.
    /// </summary>
    public IObservable<Tensor> Process<T>(IObservable<T> source)
    {
        var device = Device ?? CPU;
        var tensor = _values.to(device).clone();
        return source.Select(_ => tensor);
    }
}
