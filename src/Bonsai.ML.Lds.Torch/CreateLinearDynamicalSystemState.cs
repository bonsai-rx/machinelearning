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
[TypeConverter(typeof(TensorOperatorConverter))]
public class CreateLinearDynamicalSystemState : IScalarTypeProvider
{
    /// <inheritdoc/>
    [Description("The data type of the tensor elements.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// The device on which to create the tensor.
    /// </summary>
    [Description("The device on which to create the tensor.")]
    [XmlIgnore]
    public Device Device { get; set; }

    /// <summary>
    /// The mean of the state.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Mean
    {
        get => _mean;
        set => _mean = value;
    }

    /// <summary>
    /// The XML string representation of the mean for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Mean))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string MeanXml
    {
        get => TensorConverter.ConvertToString(_mean, Type);
        set => _mean = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The covariance of the state.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Covariance
    {
        get => _covariance;
        set => _covariance = value;
    }

    /// <summary>
    /// The XML string representation of the covariance for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Covariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string CovarianceXml
    {
        get => TensorConverter.ConvertToString(_covariance, Type);
        set => _covariance = TensorConverter.ConvertFromString(value, Type);
    }

    private Tensor _mean = null;
    private Tensor _covariance = null;

    /// <summary>
    /// Creates an observable sequence and emits the state for a linear gaussian dynamical system.
    /// </summary>
    /// <returns></returns>
    public IObservable<LinearDynamicalSystemState> Process()
    {
        return Observable.Defer(() =>
        {
            var device = Device ?? CPU;
            var mean = _mean?.to(device) ?? throw new InvalidOperationException("The mean of the state must be specified.");
            var covariance = _covariance?.to(device) ?? throw new InvalidOperationException("The covariance of the state must be specified.");
            return Observable.Return(new LinearDynamicalSystemState(mean, covariance));
        });
    }

    /// <summary>
    /// Processes an observable sequence and emits new states for a linear gaussian dynamical system.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public IObservable<LinearDynamicalSystemState> Process<T>(IObservable<T> source)
    {
        return source.Select(_ =>
        {
            var device = Device ?? CPU;
            var mean = _mean?.to(device) ?? throw new InvalidOperationException("The mean of the state must be specified.");
            var covariance = _covariance?.to(device) ?? throw new InvalidOperationException("The covariance of the state must be specified.");
            return new LinearDynamicalSystemState(mean, covariance);
        });
    }

    /// <summary>
    /// Processes an observable sequence of a tuple of tensors (mean and covariance) and emits a state for a linear gaussian dynamical system.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<LinearDynamicalSystemState> Process(IObservable<Tuple<Tensor, Tensor>> source)
    {
        return source.Select(input =>
        {
            return new LinearDynamicalSystemState(input.Item1, input.Item2);
        });
    }
}