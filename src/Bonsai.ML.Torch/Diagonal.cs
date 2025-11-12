using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch;

/// <summary>
/// Creates a diagonal matrix. If input is a 1D tensor, it creates a diagonal matrix with the elements of the tensor on the diagonal.
/// If input is a 2D tensor, it returns the diagonal elements as a 1D tensor.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a diagonal matrix with the given data type, size, and value.")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class Diagonal : IScalarTypeProvider
{
    private Tensor _values;
    
    /// <summary>
    /// Gets or sets the values to include in the diagonal.
    /// </summary>
    [Description("The values to include in the diagonal.")]
    [TypeConverter(typeof(TensorConverter))]
    [XmlIgnore]
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
        get => TensorConverter.ConvertToString(_values, Type);
        set => _values = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Gets or sets the data type of the tensor elements.
    /// </summary>
    [Description("The data type of the tensor elements.")]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Gets or sets the device on which to create the tensor.
    /// </summary>
    [Description("The device on which to create the tensor.")]
    [XmlIgnore]
    public Device Device { get; set; } = null;

    /// <summary>
    /// Gets or sets the diagonal offset. Default is 0, which means the main diagonal.
    /// </summary>
    [Description("The diagonal offset. Default is 0, which means the main diagonal.")]
    public int Offset { get; set; } = 0;

    /// <summary>
    /// Creates an observable sequence containing a single diagonal matrix constructed from the specified data type, size and values.
    /// </summary>
    public IObservable<Tensor> Process()
    {
        var device = Device ?? CPU;
        var inputTensor = _values.to(device);
        return Observable.Return(diag(inputTensor, Offset));
    }

    /// <summary>
    /// Generates an observable sequence of tensors by extracting the diagonal of the input.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        var device = Device ?? CPU;
        return source.Select(value => _values is not null 
            ? diag(_values, Offset).to(device) 
            : diag(value, Offset).to(device));
    }

    /// <summary>
    /// Generates an observable sequence of tensors by extracting the diagonal of the input.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process<T>(IObservable<T> source)
    {
        var device = Device ?? CPU;
        var inputTensor = _values.to(device);
        return source.Select(value => diag(inputTensor, Offset));
    }
}
