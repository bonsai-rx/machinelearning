using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Represents the result of an expectation-maximization step for a Kalman filter model.
/// </summary>
/// <param name="logLikelihood"></param>
/// <param name="parameters"></param>
public struct ExpectationMaximizationResult(
    Tensor logLikelihood,
    KalmanFilterParameters parameters)
{
    /// <summary>
    /// The log likelihood of the observed data given the model parameters after each iteration.
    /// </summary>
    public Tensor LogLikelihood = logLikelihood;

    /// <summary>
    /// The final updated Kalman filter parameters after the last expectation-maximization step.
    /// </summary>
    public KalmanFilterParameters Parameters = parameters;
}