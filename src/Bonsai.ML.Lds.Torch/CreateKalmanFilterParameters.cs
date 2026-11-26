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
    private Tensor _transitionMatrix;
    private Tensor _measurementFunction;
    private Tensor _processNoiseCovariance;
    private Tensor _measurementNoiseCovariance;
    private Tensor _initialMean;
    private Tensor _initialCovariance;
    private Tensor _stateOffset;
    private Tensor _observationOffset;

    /// <inheritdoc/>
    [Description("The data type of the tensor elements.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Gets or sets the device on which to create the tensor.
    /// </summary>
    [XmlIgnore]
    [Description("The device on which to create the tensor.")]
    public Device Device { get; set; }

    /// <summary>
    /// Gets or sets the number of states in the Kalman filter model.
    /// </summary>
    [Description("The number of states in the model. If not specified, it is inferred from the parameters.")]
    public int? NumStates { get; set; }

    /// <summary>
    /// Gets or sets the number of observations in the Kalman filter model.
    /// </summary>
    [Description("The number of observations in the model. If not specified, it is inferred from the parameters.")]
    public int? NumObservations { get; set; }

    /// <summary>
    /// Gets or sets the state transition matrix.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("The state transition matrix. If not specified, the identity matrix is used.")]
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
    /// Gets or sets the measurement function.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("The measurement function that maps states to observations. If not specified, the identity matrix is used.")]
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
    /// Gets or sets the process noise variance.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("The process noise covariance, given as a matrix or as a scalar variance of every state. If not specified, a variance of 1 is used.")]
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
    /// Gets or sets the measurement noise covariance matrix.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("The measurement noise covariance, given as a matrix or as a scalar variance of every observation. If not specified, a variance of 1 is used.")]
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
    /// Gets or sets the initial mean.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("The mean of the initial state. If not specified, it is zero.")]
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
    /// Gets or sets the initial covariance.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("The covariance of the initial state. If not specified, the identity matrix is used.")]
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
    /// Gets or sets the state offset.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("The constant offset added to the state at each time step. If not specified while the observation offset is, it is zero.")]
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
    /// Gets or sets the observation offset.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Description("The constant offset added to each observation. If not specified while the state offset is, it is zero.")]
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
    /// <returns>A sequence containing the Kalman filter parameters.</returns>
    public IObservable<KalmanFilterParameters> Process()
    {
        return Observable.Defer(() => Observable.Return(new KalmanFilterParameters(
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
        )));
    }

    /// <summary>
    /// Creates parameters for a Kalman filter model for each element in the input sequence.
    /// </summary>
    /// <typeparam name="T">The type of the elements in the source sequence.</typeparam>
    /// <param name="source">The sequence of notifications used to create new parameters.</param>
    /// <returns>A sequence of Kalman filter parameters, one for each element in the source sequence.</returns>
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
