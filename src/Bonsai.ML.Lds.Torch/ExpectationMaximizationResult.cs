using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Represents the result of an expectation-maximization step for a Kalman filter model.
/// </summary>
/// <param name="logLikelihood">The log likelihood after each iteration.</param>
/// <param name="parameters">The parameters after the latest iteration.</param>
/// <param name="finished">Whether the algorithm has finished.</param>
public struct ExpectationMaximizationResult(
    Tensor logLikelihood,
    KalmanFilterParameters parameters,
    bool finished = false)
{
    /// <summary>
    /// The log likelihood of the observed data given the model parameters after each iteration.
    /// </summary>
    public Tensor LogLikelihood = logLikelihood;

    /// <summary>
    /// The final updated Kalman filter parameters after the last expectation-maximization step.
    /// </summary>
    public KalmanFilterParameters Parameters = parameters;

    /// <summary>
    /// Indicates whether the EM algorithm has finished.
    /// </summary>
    public bool Finished = finished;
}