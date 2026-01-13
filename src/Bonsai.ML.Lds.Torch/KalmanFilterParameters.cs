using System;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Represents the parameters of a Kalman filter model.
/// </summary>
public class KalmanFilterParameters : nn.Module
{
    private readonly static StringBuilder _sb = new();
    private readonly ScalarType _scalarType;
    private readonly Device _device;

    /// <summary>
    /// The number of states in the system.
    /// </summary>
    public int NumStates { get; private set; }

    /// <summary>
    /// The number of observations in the system.
    /// </summary>
    public int NumObservations { get; private set; }

    /// <summary>
    /// The state transition matrix.
    /// </summary>
    public Tensor TransitionMatrix { get; private set; }

    /// <summary>
    /// The measurement function.
    /// </summary>
    public Tensor MeasurementFunction { get; private set; }

    /// <summary>
    /// The process noise covariance.
    /// </summary>
    public Tensor ProcessNoiseCovariance { get; private set; }

    /// <summary>
    /// The measurement noise covariance.
    /// </summary>
    public Tensor MeasurementNoiseCovariance { get; private set; }

    /// <summary>
    /// The initial mean.
    /// </summary>
    public Tensor InitialMean { get; private set; }

    /// <summary>
    /// The initial covariance.
    /// </summary>
    public Tensor InitialCovariance { get; private set; }

    /// <summary>
    /// The optional state offset.
    /// </summary>
    public Tensor StateOffset { get; private set; }

    /// <summary>
    /// The optional observation offset.
    /// </summary>
    public Tensor ObservationOffset { get; private set; }

    /// <summary>
    /// Indicates whether any offsets have been provided.
    /// </summary>
    public bool OffsetsProvided => StateOffset is not null || ObservationOffset is not null;

    /// <summary>
    /// The data type of the tensors.
    /// </summary>
    public ScalarType ScalarType => _scalarType;

    /// <summary>
    /// The device on which the tensors are allocated.
    /// </summary>
    public Device Device => _device;

    /// <summary>
    /// Initializes a new instance of the <see cref="KalmanFilterParameters"/> class with the specified parameters.
    /// </summary>
    /// <param name="numStates"></param>
    /// <param name="numObservations"></param>
    /// <param name="transitionMatrix"></param>
    /// <param name="measurementFunction"></param>
    /// <param name="processNoiseCovariance"></param>
    /// <param name="measurementNoiseCovariance"></param>
    /// <param name="initialMean"></param>
    /// <param name="initialCovariance"></param>
    /// <param name="stateOffset"></param>
    /// <param name="observationOffset"></param>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    /// <param name="requiresGrad"></param>
    public KalmanFilterParameters(
        int? numStates = null,
        int? numObservations = null,
        Tensor transitionMatrix = null,
        Tensor measurementFunction = null,
        Tensor processNoiseCovariance = null,
        Tensor measurementNoiseCovariance = null,
        Tensor initialMean = null,
        Tensor initialCovariance = null,
        Tensor stateOffset = null,
        Tensor observationOffset = null,
        Device device = null,
        ScalarType? scalarType = null,
        bool requiresGrad = false) : base("KalmanFilterParameters")
    {
        numStates ??= InferNumStates(transitionMatrix, measurementFunction, initialMean, initialCovariance, processNoiseCovariance, stateOffset);
        numObservations ??= InferNumObservations(measurementFunction, measurementNoiseCovariance, observationOffset);

        if (numStates is null)
            throw new ArgumentOutOfRangeException(nameof(numStates), "Number of states must be specified or inferred from the parameters.");
        if (numObservations is null)
            throw new ArgumentOutOfRangeException(nameof(numObservations), "Number of observations must be specified or inferred from the parameters.");

        transitionMatrix = transitionMatrix?.clone() ?? eye(numStates.Value);
        measurementFunction = measurementFunction?.clone() ?? eye(numObservations.Value, numStates.Value);
        initialMean = initialMean?.clone() ?? zeros(numStates.Value);
        initialCovariance = initialCovariance?.clone() ?? eye(numStates.Value);

        processNoiseCovariance = processNoiseCovariance?.NumberOfElements == 1
            ? CreateCovarianceMatrixFromScalar(processNoiseCovariance, numStates.Value, "Process noise variance")
            : processNoiseCovariance?.clone()
            ?? CreateCovarianceMatrixFromScalar(1.0, numStates.Value, "Process noise variance");

        measurementNoiseCovariance = measurementNoiseCovariance?.NumberOfElements == 1
            ? CreateCovarianceMatrixFromScalar(measurementNoiseCovariance, numObservations.Value, "Measurement noise variance")
            : measurementNoiseCovariance?.clone()
            ?? CreateCovarianceMatrixFromScalar(1.0, numObservations.Value, "Measurement noise variance");

        NumStates = numStates.Value;
        NumObservations = numObservations.Value;
        TransitionMatrix = transitionMatrix;
        MeasurementFunction = measurementFunction;
        ProcessNoiseCovariance = processNoiseCovariance;
        MeasurementNoiseCovariance = measurementNoiseCovariance;
        InitialMean = initialMean;
        InitialCovariance = initialCovariance;
        StateOffset = stateOffset;
        ObservationOffset = observationOffset;

        Validate();

        if (device is not null)
            this.to(device);
        if (scalarType is not null)
            this.to(scalarType.Value);

        _device = TransitionMatrix.device;
        _scalarType = TransitionMatrix.dtype;

        SetGrad(requiresGrad);
    }

    /// <summary>
    /// Validates the Kalman filter parameters.
    /// </summary>
    private void Validate()
    {
        int numStates = InferNumStates(TransitionMatrix, MeasurementFunction, InitialMean, InitialCovariance, ProcessNoiseCovariance, StateOffset);

        int numObservations = InferNumObservations(MeasurementFunction, MeasurementNoiseCovariance, ObservationOffset);

        ValidateMatrix(TransitionMatrix, "Transition matrix", isSquare: true, expectedDimension1: numStates);

        ValidateMatrix(MeasurementFunction, "Measurement function", expectedDimension1: numObservations, expectedDimension2: numStates);

        ValidateMatrix(ProcessNoiseCovariance, "Process noise covariance", isSquare: true, expectedDimension1: numStates);

        ValidateMatrix(MeasurementNoiseCovariance, "Measurement noise covariance", isSquare: true, expectedDimension1: numObservations);

        ValidateVector(InitialMean, "Initial mean", numStates);
        ValidateMatrix(InitialCovariance, "Initial covariance", isSquare: true, expectedDimension1: numStates);

        if (StateOffset is not null)
            ValidateVector(StateOffset, "State offset", numStates);

        if (ObservationOffset is not null)
            ValidateVector(ObservationOffset, "Observation offset", numObservations);

        NumStates = numStates;
        NumObservations = numObservations;
    }

    /// <summary>
    /// Creates a copy of the current Kalman filter parameters.
    /// </summary>
    /// <returns></returns>
    public KalmanFilterParameters Copy() => new(
        NumStates,
        NumObservations,
        TransitionMatrix?.clone(),
        MeasurementFunction?.clone(),
        ProcessNoiseCovariance?.clone(),
        MeasurementNoiseCovariance?.clone(),
        InitialMean?.clone(),
        InitialCovariance?.clone(),
        StateOffset?.clone(),
        ObservationOffset?.clone()
    );

    /// <summary>
    /// Sets the requires_grad flag for all tensors in the Kalman filter parameters.
    /// </summary>
    /// <param name="requiresGrad"></param>
    public void SetGrad(bool requiresGrad)
    {
        TransitionMatrix = TransitionMatrix?.requires_grad_(requiresGrad);
        MeasurementFunction = MeasurementFunction?.requires_grad_(requiresGrad);
        ProcessNoiseCovariance = ProcessNoiseCovariance?.requires_grad_(requiresGrad);
        MeasurementNoiseCovariance = MeasurementNoiseCovariance?.requires_grad_(requiresGrad);
        InitialMean = InitialMean?.requires_grad_(requiresGrad);
        InitialCovariance = InitialCovariance?.requires_grad_(requiresGrad);
        StateOffset = StateOffset?.requires_grad_(requiresGrad);
        ObservationOffset = ObservationOffset?.requires_grad_(requiresGrad);
    }

    private static int InferNumStates(Tensor transitionMatrix, Tensor measurementFunction, Tensor initialMean, Tensor initialCovariance, Tensor processNoiseCovariance, Tensor stateOffset)
    {
        if (transitionMatrix is not null)
        {
            ValidateMatrix(transitionMatrix, "Transition matrix", isSquare: true);
            return (int)transitionMatrix.size(0);
        }
        else if (measurementFunction is not null)
        {
            ValidateMatrix(measurementFunction, "Measurement function");
            return (int)measurementFunction.size(1);
        }
        else if (initialMean is not null)
        {
            ValidateVector(initialMean, "Initial mean");
            return (int)initialMean.size(0);
        }
        else if (initialCovariance is not null)
        {
            ValidateMatrix(initialCovariance, "Initial covariance", isSquare: true);
            return (int)initialCovariance.size(0);
        }
        else if (processNoiseCovariance is not null)
        {
            ValidateMatrix(processNoiseCovariance, "Process noise covariance", isSquare: true);
            return (int)processNoiseCovariance.size(0);
        }
        else if (stateOffset is not null)
        {
            ValidateVector(stateOffset, "State offset");
            return (int)stateOffset.size(0);
        }
        else
        {
            throw new ArgumentException("At least one of the parameters must be provided to infer the number of states.");
        }
    }

    private static int InferNumObservations(Tensor measurementFunction, Tensor measurementNoiseCovariance, Tensor observationOffset)
    {
        if (measurementFunction is not null)
        {
            ValidateMatrix(measurementFunction, "Measurement function");
            return (int)measurementFunction.size(0);
        }
        else if (measurementNoiseCovariance is not null)
        {
            ValidateMatrix(measurementNoiseCovariance, "Measurement noise covariance", isSquare: true);
            return (int)measurementNoiseCovariance.size(0);
        }
        else if (observationOffset is not null)
        {
            ValidateVector(observationOffset, "Observation offset");
            return (int)observationOffset.size(0);
        }
        else
        {
            throw new ArgumentException("At least one of the measurement function or measurement noise covariance must be provided to infer the number of observations.");
        }
    }

    private static void ValidateMatrix(Tensor matrix, string name, bool isSquare = false, int? expectedDimension1 = null, int? expectedDimension2 = null)
    {
        if (matrix.NumberOfElements == 0)
            throw new ArgumentException($"{name} must be a non-empty matrix.");

        if (matrix.Dimensions != 2)
            throw new ArgumentException($"{name} must be 2-dimensional.");

        if (isSquare && matrix.size(0) != matrix.size(1))
            throw new ArgumentException($"{name} must be square.");

        if (expectedDimension1.HasValue && matrix.size(0) != expectedDimension1.Value)
            throw new ArgumentException($"{name} must have {expectedDimension1.Value} rows.");

        if (expectedDimension2.HasValue && matrix.size(1) != expectedDimension2.Value)
            throw new ArgumentException($"{name} must have {expectedDimension2.Value} columns.");
    }

    private static void ValidateVector(Tensor vector, string name, int? expectedLength = null)
    {
        if (vector.NumberOfElements == 0)
            throw new ArgumentException($"{name} must be a non-empty vector.");

        if (vector.Dimensions != 1)
            throw new ArgumentException($"{name} must be a vector.");

        if (expectedLength.HasValue && vector.NumberOfElements != expectedLength.Value)
            throw new ArgumentException($"{name} must be a vector with length equal to {expectedLength.Value}.");
    }

    private static void ValidateScalar(Tensor scalar, string name)
    {
        if (scalar.NumberOfElements != 1)
            throw new ArgumentException($"{name} must be a scalar.");
    }

    private static Tensor CreateCovarianceMatrixFromScalar(Tensor variance, int dimension, string name)
    {
        ValidateScalar(variance, name);
        var scalar = variance.clone().squeeze();
        return scalar * eye(dimension);
    }

    /// <inheritdoc/>
    public override string ToString() => _sb.Length == 0 ? _sb.Append(
        $"KalmanFilterParameters(NumStates={NumStates}, NumObservations={NumObservations}, TransitionMatrix={TransitionMatrix}, MeasurementFunction={MeasurementFunction}, ProcessNoiseCovariance={ProcessNoiseCovariance}, MeasurementNoiseCovariance={MeasurementNoiseCovariance}, InitialMean={InitialMean}, InitialCovariance={InitialCovariance})" + (StateOffset is not null ? $", StateOffset={StateOffset}" : "") + (ObservationOffset is not null ? $", ObservationOffset={ObservationOffset}" : "")).ToString() : _sb.ToString();

    /// <summary>
    /// Returns a string representation of the Kalman filter parameters with the specified tensor string style.
    /// </summary>
    /// <param name="tensorStringStyle"></param>
    /// <returns></returns>
    public string ToString(TensorStringStyle tensorStringStyle) => _sb.Length == 0 ? _sb.Append(
        $"KalmanFilterParameters(NumStates={NumStates}, NumObservations={NumObservations}, TransitionMatrix={TransitionMatrix.ToString(tensorStringStyle)}, MeasurementFunction={MeasurementFunction.ToString(tensorStringStyle)}, ProcessNoiseCovariance={ProcessNoiseCovariance.ToString(tensorStringStyle)}, MeasurementNoiseCovariance={MeasurementNoiseCovariance.ToString(tensorStringStyle)}, InitialMean={InitialMean.ToString(tensorStringStyle)}, InitialCovariance={InitialCovariance.ToString(tensorStringStyle)})" + (StateOffset is not null ? $", StateOffset={StateOffset.ToString(tensorStringStyle)}" : "") + (ObservationOffset is not null ? $", ObservationOffset={ObservationOffset.ToString(tensorStringStyle)}" : "")).ToString() : _sb.ToString();
}
