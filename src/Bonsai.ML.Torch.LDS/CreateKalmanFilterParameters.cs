using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Initializes the parameters for a new Kalman filter model.
/// </summary>
[Combinator]
[Description("Initializes the parameters for a new Kalman filter model.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CreateKalmanFilterParameters : IScalarTypeProvider
{
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
    private ScalarType _scalarType = ScalarType.Float32;

    private Tensor _transitionMatrix = null;
    /// <summary>
    /// The state transition matrix.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor TransitionMatrix
    {
        get => _transitionMatrix;
        set => _transitionMatrix = value?.to_type(Type);
    }

    /// <summary>
    /// The XML string representation of the transition matrix for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(TransitionMatrix))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string TransitionMatrixXml
    {
        get => TensorConverter.ConvertToString(TransitionMatrix, _scalarType);
        set => TransitionMatrix = TensorConverter.ConvertFromString(value, _scalarType);
    }

    private Tensor _measurementFunction = null;
    /// <summary>
    /// The measurement function.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor MeasurementFunction
    {
        get => _measurementFunction;
        set => _measurementFunction = value?.to_type(Type);
    }

    /// <summary>
    /// The XML string representation of the measurement function for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(MeasurementFunction))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string MeasurementFunctionXml
    {
        get => TensorConverter.ConvertToString(MeasurementFunction, _scalarType);
        set => MeasurementFunction = TensorConverter.ConvertFromString(value, _scalarType);
    }

    private Tensor _processNoiseCovariance = null;
    /// <summary>
    /// The process noise variance.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor ProcessNoiseCovariance
    {
        get => _processNoiseCovariance;
        set => _processNoiseCovariance = value?.to_type(Type);
    }

    /// <summary>
    /// The XML string representation of the process noise variance for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(ProcessNoiseCovariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ProcessNoiseCovarianceXml
    {
        get => TensorConverter.ConvertToString(ProcessNoiseCovariance, _scalarType);
        set => ProcessNoiseCovariance = TensorConverter.ConvertFromString(value, _scalarType);
    }

    private Tensor _measurementNoiseCovariance = null;
    /// <summary>
    /// The measurement noise covariance matrix.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor MeasurementNoiseCovariance
    {
        get => _measurementNoiseCovariance;
        set => _measurementNoiseCovariance = value?.to_type(Type);
    }

    /// <summary>
    /// The XML string representation of the measurement noise covariance matrix for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(MeasurementNoiseCovariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string MeasurementNoiseCovarianceXml
    {
        get => TensorConverter.ConvertToString(MeasurementNoiseCovariance, _scalarType);
        set => MeasurementNoiseCovariance = TensorConverter.ConvertFromString(value, _scalarType);
    }

    private Tensor _initialMean = null;
    /// <summary>
    /// The initial mean.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor InitialMean
    {
        get => _initialMean;
        set => _initialMean = value?.to_type(Type);
    }

    /// <summary>
    /// The XML string representation of the initial state for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(InitialMean))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string InitialMeanXml
    {
        get => TensorConverter.ConvertToString(InitialMean, _scalarType);
        set => InitialMean = TensorConverter.ConvertFromString(value, _scalarType);
    }

    private Tensor _initialCovariance = null;
    /// <summary>
    /// The initial covariance.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor InitialCovariance
    {
        get => _initialCovariance;
        set => _initialCovariance = value?.to_type(Type);
    }

    /// <summary>
    /// The XML string representation of the initial covariance for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(InitialCovariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string InitialCovarianceXml
    {
        get => TensorConverter.ConvertToString(InitialCovariance, _scalarType);
        set => InitialCovariance = TensorConverter.ConvertFromString(value, _scalarType);
    }

    private void ConvertTensorsScalarType(ScalarType scalarType)
    {
        _transitionMatrix = _transitionMatrix?.to_type(scalarType);
        _measurementFunction = _measurementFunction?.to_type(scalarType);
        _processNoiseCovariance = _processNoiseCovariance?.to_type(scalarType);
        _measurementNoiseCovariance = _measurementNoiseCovariance?.to_type(scalarType);
        _initialMean = _initialMean?.to_type(scalarType);
        _initialCovariance = _initialCovariance?.to_type(scalarType);
    }

    /// <summary>
    /// Creates parameters for a Kalman filter model using the properties of this class.
    /// </summary>
    public IObservable<KalmanFilterParameters> Process()
    {
        var parameters = new KalmanFilterParameters(
            transitionMatrix: _transitionMatrix,
            measurementFunction: _measurementFunction,
            processNoiseCovariance: _processNoiseCovariance,
            measurementNoiseCovariance: _measurementNoiseCovariance,
            initialMean: _initialMean,
            initialCovariance: _initialCovariance
        );

        return Observable.Return(parameters);
    }

    /// <summary>
    /// Creates parameters for a Kalman filter model for each element in the input sequence.
    /// </summary>
    public IObservable<KalmanFilterParameters> Process<T>(IObservable<T> source)
    {
        return source.Select(_ =>
        {
            var parameters = new KalmanFilterParameters(
                transitionMatrix: _transitionMatrix,
                measurementFunction: _measurementFunction,
                processNoiseCovariance: _processNoiseCovariance,
                measurementNoiseCovariance: _measurementNoiseCovariance,
                initialMean: _initialMean,
                initialCovariance: _initialCovariance
            );

            return parameters;
        });
    }
}