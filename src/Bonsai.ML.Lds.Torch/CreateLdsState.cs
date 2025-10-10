using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using Bonsai.ML.Torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Creates a generic state object for a linear gaussian dynamical system.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a new state for a linear gaussian dynamical system.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CreateLdsState : IScalarTypeProvider
{
    private ScalarType _scalarType = ScalarType.Float32;
    /// <inheritdoc/>
    [Description("The data type of the tensor elements.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType Type
    {
        get => _scalarType;
        set
        {
            _scalarType = value;
            ConvertTensorsScalarType(value);
        }
    }

    /// <summary>
    /// The device on which to create the tensor.
    /// </summary>
    [Description("The device on which to create the tensor.")]
    [XmlIgnore]
    public Device Device { get; set; }

    private void ConvertTensorsScalarType(ScalarType scalarType)
    {
        _mean = _mean?.to_type(scalarType);
        _covariance = _covariance?.to_type(scalarType);
    }

    private Tensor _mean = null;
    /// <summary>
    /// The mean of the state.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Mean
    {
        get => _mean;
        set => _mean = value?.to_type(Type);
    }

    /// <summary>
    /// The XML string representation of the mean for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Mean))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string MeanXml
    {
        get => TensorConverter.ConvertToString(Mean, _scalarType);
        set => Mean = TensorConverter.ConvertFromString(value, _scalarType);
    }

    private Tensor _covariance = null;
    /// <summary>
    /// The covariance of the state.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Covariance
    {
        get => _covariance;
        set => _covariance = value?.to_type(Type);
    }

    /// <summary>
    /// The XML string representation of the covariance for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Covariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string CovarianceXml
    {
        get => TensorConverter.ConvertToString(Covariance, _scalarType);
        set => Covariance = TensorConverter.ConvertFromString(value, _scalarType);
    }

    /// <summary>
    /// Creates an observable sequence and emits the state for a linear gaussian dynamical system.
    /// </summary>
    /// <returns></returns>
    public IObservable<LdsState> Process()
    {
        return Observable.Defer(() =>
        {
            var device = Device ?? CPU;
            var mean = Mean?.to(device) ?? throw new InvalidOperationException("The mean of the state must be specified.");
            var covariance = Covariance?.to(device) ?? throw new InvalidOperationException("The covariance of the state must be specified.");
            return Observable.Return(new LdsState(mean, covariance));
        });
    }

    /// <summary>
    /// Processes an observable sequence and emits new states for a linear gaussian dynamical system.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public IObservable<LdsState> Process<T>(IObservable<T> source)
    {
        return source.Select(_ =>
        {
            var device = Device ?? CPU;
            var mean = Mean?.to(device) ?? throw new InvalidOperationException("The mean of the state must be specified.");
            var covariance = Covariance?.to(device) ?? throw new InvalidOperationException("The covariance of the state must be specified.");
            return new LdsState(mean, covariance);
        });
    }
}