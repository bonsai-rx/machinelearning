using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using Bonsai.ML.Torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Initializes the parameters for a new Kalman filter model.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Initializes the parameters for a new Kalman filter model.")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class CreateKalmanFilterParameters : IScalarTypeProvider
{
    private Tensor _transitionMatrix = null;
    private Tensor _measurementFunction = null;
    private Tensor _processNoiseCovariance = null;
    private Tensor _measurementNoiseCovariance = null;
    private Tensor _initialMean = null;
    private Tensor _initialCovariance = null;
    private Tensor _stateOffset = null;
    private Tensor _observationOffset = null;

    /// <inheritdoc/>
    [Description("The data type of the tensor elements.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// The device on which to create the tensor.
    /// </summary>
    [XmlIgnore]
    [Description("The device on which to create the tensor.")]
    public Device Device { get; set; }

    /// <summary>
    /// The number of states in the Kalman filter model.
    /// </summary>
    public int? NumStates { get; set; } = null;

    /// <summary>
    /// The number of observations in the Kalman filter model.
    /// </summary>
    public int? NumObservations { get; set; } = null;

    /// <summary>
    /// The state transition matrix.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor TransitionMatrix
    {
        get => _transitionMatrix;
        set => _transitionMatrix = value;
    }

    /// <summary>
    /// The XML string representation of the transition matrix for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(TransitionMatrix))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string TransitionMatrixXml
    {
        get => TensorConverter.ConvertToString(_transitionMatrix, Type);
        set => _transitionMatrix = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The measurement function.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor MeasurementFunction
    {
        get => _measurementFunction;
        set => _measurementFunction = value;
    }

    /// <summary>
    /// The XML string representation of the measurement function for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(MeasurementFunction))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string MeasurementFunctionXml
    {
        get => TensorConverter.ConvertToString(_measurementFunction, Type);
        set => _measurementFunction = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The process noise variance.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor ProcessNoiseCovariance
    {
        get => _processNoiseCovariance;
        set => _processNoiseCovariance = value;
    }

    /// <summary>
    /// The XML string representation of the process noise variance for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(ProcessNoiseCovariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ProcessNoiseCovarianceXml
    {
        get => TensorConverter.ConvertToString(_processNoiseCovariance, Type);
        set => _processNoiseCovariance = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The measurement noise covariance matrix.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor MeasurementNoiseCovariance
    {
        get => _measurementNoiseCovariance;
        set => _measurementNoiseCovariance = value;
    }

    /// <summary>
    /// The XML string representation of the measurement noise covariance matrix for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(MeasurementNoiseCovariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string MeasurementNoiseCovarianceXml
    {
        get => TensorConverter.ConvertToString(_measurementNoiseCovariance, Type);
        set => _measurementNoiseCovariance = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The initial mean.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor InitialMean
    {
        get => _initialMean;
        set => _initialMean = value;
    }

    /// <summary>
    /// The XML string representation of the initial state for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(InitialMean))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string InitialMeanXml
    {
        get => TensorConverter.ConvertToString(_initialMean, Type);
        set => _initialMean = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The initial covariance.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor InitialCovariance
    {
        get => _initialCovariance;
        set => _initialCovariance = value;
    }

    /// <summary>
    /// The XML string representation of the initial covariance for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(InitialCovariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string InitialCovarianceXml
    {
        get => TensorConverter.ConvertToString(_initialCovariance, Type);
        set => _initialCovariance = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The state offset.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor StateOffset
    {
        get => _stateOffset;
        set => _stateOffset = value;
    }

    /// <summary>
    /// The XML string representation of the state offset for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(StateOffset))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string StateOffsetXml
    {
        get => TensorConverter.ConvertToString(_stateOffset, Type);
        set => _stateOffset = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The observation offset.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor ObservationOffset
    {
        get => _observationOffset;
        set => _observationOffset = value;
    }

    /// <summary>
    /// The XML string representation of the observation offset for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(ObservationOffset))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ObservationOffsetXml
    {
        get => TensorConverter.ConvertToString(_observationOffset, Type);
        set => _observationOffset = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// Creates parameters for a Kalman filter model using the properties of this class.
    /// </summary>
    public IObservable<KalmanFilterParameters> Process()
    {
        return Observable.Return(new KalmanFilterParameters(
            numStates: NumStates,
            numObservations: NumObservations,
            transitionMatrix: _transitionMatrix,
            measurementFunction: _measurementFunction,
            processNoiseCovariance: _processNoiseCovariance,
            measurementNoiseCovariance: _measurementNoiseCovariance,
            initialMean: _initialMean,
            initialCovariance: _initialCovariance,
            stateOffset: _stateOffset,
            observationOffset: _observationOffset,
            scalarType: Type,
            device: Device
        ));
    }

    /// <summary>
    /// Creates parameters for a Kalman filter model for each element in the input sequence.
    /// </summary>
    public IObservable<KalmanFilterParameters> Process<T>(IObservable<T> source)
    {
        return source.Select(_ =>
        {
            return new KalmanFilterParameters(
                numStates: NumStates,
                numObservations: NumObservations,
                transitionMatrix: _transitionMatrix,
                measurementFunction: _measurementFunction,
                processNoiseCovariance: _processNoiseCovariance,
                measurementNoiseCovariance: _measurementNoiseCovariance,
                initialMean: _initialMean,
                initialCovariance: _initialCovariance,
                stateOffset: _stateOffset,
                observationOffset: _observationOffset,
                scalarType: Type,
                device: Device
            );
        });
    }
}
