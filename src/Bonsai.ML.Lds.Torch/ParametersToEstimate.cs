namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Represents which parameters of a Kalman filter model to estimate.
/// </summary>
/// <remarks>
/// Initializes a new instance of the <see cref="ParametersToEstimate"/> class. Every parameter
/// except the offsets is estimated unless specified otherwise.
/// </remarks>
/// <param name="transitionMatrix">Whether to estimate the state transition matrix.</param>
/// <param name="measurementFunction">Whether to estimate the measurement function.</param>
/// <param name="processNoiseCovariance">Whether to estimate the process noise covariance.</param>
/// <param name="measurementNoiseCovariance">
/// Whether to estimate the measurement noise covariance.
/// </param>
/// <param name="initialMean">Whether to estimate the initial mean.</param>
/// <param name="initialCovariance">Whether to estimate the initial covariance.</param>
/// <param name="stateOffset">Whether to estimate the state offset.</param>
/// <param name="observationOffset">Whether to estimate the observation offset.</param>
public class ParametersToEstimate(
    bool transitionMatrix = true,
    bool measurementFunction = true,
    bool processNoiseCovariance = true,
    bool measurementNoiseCovariance = true,
    bool initialMean = true,
    bool initialCovariance = true,
    bool stateOffset = false,
    bool observationOffset = false)
{
    /// <summary>
    /// Gets a value indicating whether to estimate the state transition matrix.
    /// </summary>
    public bool TransitionMatrix { get; } = transitionMatrix;

    /// <summary>
    /// Gets a value indicating whether to estimate the measurement function.
    /// </summary>
    public bool MeasurementFunction { get; } = measurementFunction;

    /// <summary>
    /// Gets a value indicating whether to estimate the process noise covariance.
    /// </summary>
    public bool ProcessNoiseCovariance { get; } = processNoiseCovariance;

    /// <summary>
    /// Gets a value indicating whether to estimate the measurement noise covariance.
    /// </summary>
    public bool MeasurementNoiseCovariance { get; } = measurementNoiseCovariance;

    /// <summary>
    /// Gets a value indicating whether to estimate the initial mean.
    /// </summary>
    public bool InitialMean { get; } = initialMean;

    /// <summary>
    /// Gets a value indicating whether to estimate the initial covariance.
    /// </summary>
    public bool InitialCovariance { get; } = initialCovariance;

    /// <summary>
    /// Gets a value indicating whether to estimate the state offset.
    /// </summary>
    public bool StateOffset { get; } = stateOffset;

    /// <summary>
    /// Gets a value indicating whether to estimate the observation offset.
    /// </summary>
    public bool ObservationOffset { get; } = observationOffset;
}
