using System;
using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Represents the parameters of a Kalman filter model.
/// </summary>
/// <remarks>
/// Initializes a new instance of the <see cref="KalmanFilterParameters"/> struct with the specified parameters.
/// </remarks>
/// <param name="numStates"></param>
/// <param name="numObservations"></param>
/// <param name="transitionMatrix"></param>
/// <param name="measurementFunction"></param>
/// <param name="processNoiseCovariance"></param>
/// <param name="measurementNoiseCovariance"></param>
/// <param name="initialMean"></param>
/// <param name="initialCovariance"></param>
/// <param name="device"></param>
/// <param name="scalarType"></param>
/// <param name="requiresGrad"></param>
/// <param name="validated"></param>
public struct KalmanFilterParameters(
    int numStates,
    int numObservations,
    Tensor transitionMatrix = null,
    Tensor measurementFunction = null,
    Tensor processNoiseCovariance = null,
    Tensor measurementNoiseCovariance = null,
    Tensor initialMean = null,
    Tensor initialCovariance = null,
    Device device = null,
    ScalarType? scalarType = null,
    bool requiresGrad = false,
    bool validated = false)
{
    /// <summary>
    /// The number of states in the system.
    /// </summary>
    public int NumStates = numStates;

    /// <summary>
    /// The number of observations in the system.
    /// </summary>
    public int NumObservations = numObservations;

    /// <summary>
    /// The state transition matrix.
    /// </summary>
    public Tensor TransitionMatrix = transitionMatrix;

    /// <summary>
    /// The measurement function.
    /// </summary>
    public Tensor MeasurementFunction = measurementFunction;

    /// <summary>
    /// The process noise covariance.
    /// </summary>
    public Tensor ProcessNoiseCovariance = processNoiseCovariance;

    /// <summary>
    /// The measurement noise covariance.
    /// </summary>
    public Tensor MeasurementNoiseCovariance = measurementNoiseCovariance;

    /// <summary>
    /// The initial mean.
    /// </summary>
    public Tensor InitialMean = initialMean;

    /// <summary>
    /// The initial covariance.
    /// </summary>
    public Tensor InitialCovariance = initialCovariance;

    /// <summary>
    /// The device to use for tensor operations.
    /// </summary>
    public Device Device = device ?? CPU;

    /// <summary>
    /// The data type of the tensors.
    /// </summary>
    public ScalarType ScalarType = scalarType ?? ScalarType.Float32;

    /// <summary>
    /// Indicates whether the tensors require gradient computation.
    /// </summary>
    public bool RequiresGrad = requiresGrad;

    /// <summary>
    /// Indicates whether the parameters have been validated.
    /// </summary>
    /// <remarks>
    /// This field is used to avoid redundant validation checks.
    /// </remarks>
    public bool Validated = validated;

    /// <summary>
    /// Initializes the Kalman filter parameters.
    /// </summary>
    /// <param name="numStates"></param>
    /// <param name="numObservations"></param>
    /// <param name="transitionMatrix"></param>
    /// <param name="measurementFunction"></param>
    /// <param name="processNoiseCovariance"></param>
    /// <param name="measurementNoiseCovariance"></param>
    /// <param name="initialMean"></param>
    /// <param name="initialCovariance"></param>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    /// <param name="requiresGrad"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static KalmanFilterParameters Initialize(
        int? numStates = null,
        int? numObservations = null,
        Tensor transitionMatrix = null,
        Tensor measurementFunction = null,
        Tensor processNoiseCovariance = null,
        Tensor measurementNoiseCovariance = null,
        Tensor initialMean = null,
        Tensor initialCovariance = null,
        Device device = null,
        ScalarType? scalarType = null,
        bool requiresGrad = false
    )
    {
        var trueNumStates = numStates ?? -1;
        var trueNumObservations = numObservations ?? -1;

        if (numStates is null)
        {
            ValidateNumStates(transitionMatrix, measurementFunction, initialMean, initialCovariance, processNoiseCovariance, out trueNumStates);
        }

        if (trueNumStates <= 0)
            throw new ArgumentOutOfRangeException(nameof(trueNumStates), "Number of states must be greater than zero.");

        if (numObservations is null)
        {
            ValidateNumObservations(measurementFunction, measurementNoiseCovariance, out trueNumObservations);
        }

        if (trueNumObservations <= 0)
            throw new ArgumentOutOfRangeException(nameof(numObservations), "Number of observations must be greater than zero.");

        transitionMatrix = transitionMatrix?.clone() ?? eye(trueNumStates);
        measurementFunction = measurementFunction?.clone() ?? eye(trueNumObservations, trueNumStates);
        initialMean = initialMean?.clone() ?? zeros(trueNumStates);
        initialCovariance = initialCovariance?.clone() ?? eye(trueNumStates);

        processNoiseCovariance = processNoiseCovariance?.NumberOfElements == 1
            ? CreateCovarianceMatrixFromScalar(processNoiseCovariance, trueNumStates, "Process noise variance")
            : processNoiseCovariance?.clone()
            ?? CreateCovarianceMatrixFromScalar(1.0, trueNumStates, "Process noise variance");

        measurementNoiseCovariance = measurementNoiseCovariance?.NumberOfElements == 1
            ? CreateCovarianceMatrixFromScalar(measurementNoiseCovariance, trueNumObservations, "Measurement noise variance")
            : measurementNoiseCovariance?.clone()
            ?? CreateCovarianceMatrixFromScalar(1.0, trueNumObservations, "Measurement noise variance");

        var parameters = new KalmanFilterParameters(
            trueNumStates,
            trueNumObservations,
            transitionMatrix,
            measurementFunction,
            processNoiseCovariance,
            measurementNoiseCovariance,
            initialMean,
            initialCovariance,
            device,
            scalarType,
            requiresGrad
        );

        parameters.Validate();
        parameters.ToScalarType(scalarType);
        parameters.ToDevice(device);
        parameters.SetGrad(requiresGrad);

        return parameters;
    }

    /// <summary>
    /// Validates the Kalman filter parameters.
    /// </summary>
    public void Validate()
    {
        if (Validated)
            return;

        ValidateNumStates(TransitionMatrix, MeasurementFunction, InitialMean, InitialCovariance, ProcessNoiseCovariance, out NumStates);
        ValidateNumObservations(MeasurementFunction, MeasurementNoiseCovariance, out NumObservations);
        ValidateMatrix(TransitionMatrix, "Transition matrix", isSquare: true, expectedDimension1: NumStates);
        ValidateMatrix(MeasurementFunction, "Measurement function", expectedDimension1: NumObservations, expectedDimension2: NumStates);
        ValidateMatrix(ProcessNoiseCovariance, "Process noise covariance", isSquare: true, expectedDimension1: NumStates);
        ValidateMatrix(MeasurementNoiseCovariance, "Measurement noise covariance", isSquare: true, expectedDimension1: NumObservations);
        ValidateVector(InitialMean, "Initial mean", NumStates);
        ValidateMatrix(InitialCovariance, "Initial covariance", isSquare: true, expectedDimension1: NumStates);
        Validated = true;
    }

    /// <summary>
    /// Creates a copy of the current Kalman filter parameters.
    /// </summary>
    /// <returns></returns>
    public readonly KalmanFilterParameters Copy() => new(
        NumStates,
        NumObservations,
        TransitionMatrix?.clone(),
        MeasurementFunction?.clone(),
        ProcessNoiseCovariance?.clone(),
        MeasurementNoiseCovariance?.clone(),
        InitialMean?.clone(),
        InitialCovariance?.clone(),
        Device,
        ScalarType,
        RequiresGrad,
        Validated
    );

    /// <summary>
    /// Converts the tensors in the Kalman filter parameters to the specified scalar type.
    /// </summary>
    /// <param name="scalarType"></param>
    public void ToScalarType(ScalarType? scalarType)
    {
        if (scalarType is not null)
        {
            TransitionMatrix = TransitionMatrix?.to_type(scalarType.Value);
            MeasurementFunction = MeasurementFunction?.to_type(scalarType.Value);
            ProcessNoiseCovariance = ProcessNoiseCovariance?.to_type(scalarType.Value);
            MeasurementNoiseCovariance = MeasurementNoiseCovariance?.to_type(scalarType.Value);
            InitialMean = InitialMean?.to_type(scalarType.Value);
            InitialCovariance = InitialCovariance?.to_type(scalarType.Value);
        }
    }

    /// <summary>
    /// Moves the tensors in the Kalman filter parameters to the specified device.
    /// </summary>
    /// <param name="device"></param>
    public void ToDevice(Device? device)
    {
        if (device is not null)
        {
            TransitionMatrix = TransitionMatrix?.to(device);
            MeasurementFunction = MeasurementFunction?.to(device);
            ProcessNoiseCovariance = ProcessNoiseCovariance?.to(device);
            MeasurementNoiseCovariance = MeasurementNoiseCovariance?.to(device);
            InitialMean = InitialMean?.to(device);
            InitialCovariance = InitialCovariance?.to(device);
        }
    }

    /// <summary>
    /// Sets the requires_grad flag for all tensors in the Kalman filter parameters.
    /// </summary>
    /// <param name="requiresGrad"></param>
    public void SetGrad(bool requiresGrad)
    {
        if (RequiresGrad == requiresGrad)
            return;

        TransitionMatrix = TransitionMatrix?.requires_grad_(requiresGrad);
        MeasurementFunction = MeasurementFunction?.requires_grad_(requiresGrad);
        ProcessNoiseCovariance = ProcessNoiseCovariance?.requires_grad_(requiresGrad);
        MeasurementNoiseCovariance = MeasurementNoiseCovariance?.requires_grad_(requiresGrad);
        InitialMean = InitialMean?.requires_grad_(requiresGrad);
        InitialCovariance = InitialCovariance?.requires_grad_(requiresGrad);
        RequiresGrad = requiresGrad;
    }

    private static void ValidateNumStates(Tensor transitionMatrix, Tensor measurementFunction, Tensor initialMean, Tensor initialCovariance, Tensor processNoiseCovariance, out int numStates)
    {
        if (transitionMatrix is not null)
        {
            ValidateMatrix(transitionMatrix, "Transition matrix", isSquare: true);
            numStates = (int)transitionMatrix.size(0);
        }
        else if (measurementFunction is not null)
        {
            ValidateMatrix(measurementFunction, "Measurement function");
            numStates = (int)measurementFunction.size(1);
        }
        else if (initialMean is not null)
        {
            ValidateVector(initialMean, "Initial mean");
            numStates = (int)initialMean.size(0);
        }
        else if (initialCovariance is not null)
        {
            ValidateMatrix(initialCovariance, "Initial covariance", isSquare: true);
            numStates = (int)initialCovariance.size(0);
        }
        else if (processNoiseCovariance is not null)
        {
            ValidateMatrix(processNoiseCovariance, "Process noise covariance", isSquare: true);
            numStates = (int)processNoiseCovariance.size(0);
        }
        else
        {
            throw new ArgumentException("At least one of the parameters must be provided to infer the number of states.");
        }
    }

    private static void ValidateNumObservations(Tensor measurementFunction, Tensor measurementNoiseCovariance, out int numObservations)
    {
        if (measurementFunction is not null)
        {
            ValidateMatrix(measurementFunction, "Measurement function");
            numObservations = (int)measurementFunction.size(0);
        }
        else if (measurementNoiseCovariance is not null)
        {
            ValidateMatrix(measurementNoiseCovariance, "Measurement noise covariance", isSquare: true);
            numObservations = (int)measurementNoiseCovariance.size(0);
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
    public override readonly string ToString() =>
        $"KalmanFilterParameters(NumStates={NumStates}, NumObservations={NumObservations}, TransitionMatrix={TransitionMatrix}, MeasurementFunction={MeasurementFunction}, ProcessNoiseCovariance={ProcessNoiseCovariance}, MeasurementNoiseCovariance={MeasurementNoiseCovariance}, InitialMean={InitialMean}, InitialCovariance={InitialCovariance})";

    /// <summary>
    /// Returns a string representation of the Kalman filter parameters with the specified tensor string style.
    /// </summary>
    /// <param name="tensorStringStyle"></param>
    /// <returns></returns>
    public readonly string ToString(TorchSharp.TensorStringStyle tensorStringStyle) =>
        $"KalmanFilterParameters(NumStates={NumStates}, NumObservations={NumObservations}, TransitionMatrix={TransitionMatrix.ToString(tensorStringStyle)}, MeasurementFunction={MeasurementFunction.ToString(tensorStringStyle)}, ProcessNoiseCovariance={ProcessNoiseCovariance.ToString(tensorStringStyle)}, MeasurementNoiseCovariance={MeasurementNoiseCovariance.ToString(tensorStringStyle)}, InitialMean={InitialMean.ToString(tensorStringStyle)}, InitialCovariance={InitialCovariance.ToString(tensorStringStyle)})";
}