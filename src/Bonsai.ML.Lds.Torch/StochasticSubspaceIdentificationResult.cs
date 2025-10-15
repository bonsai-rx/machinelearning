using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Represents the result of subspace identification using the Kalman-Ho method.
/// </summary>
/// <param name="parameters">The identified Kalman filter parameters.</param>
/// <param name="effectiveStates">The effective states of the system determined by SVD.</param>
/// <param name="singularValues">The singular values from the SVD decomposition.</param>
public struct StochasticSubspaceIdentificationResult(
    KalmanFilterParameters parameters,
    int effectiveStates,
    Tensor singularValues)
{
    /// <summary>
    /// The identified Kalman filter parameters from subspace identification.
    /// </summary>
    public KalmanFilterParameters Parameters = parameters;

    /// <summary>
    /// The effective states of the system determined by SVD truncation.
    /// </summary>
    public int EffectiveStates = effectiveStates;

    /// <summary>
    /// The singular values from the SVD decomposition of the Hankel matrix.
    /// These can be used to assess the quality of the identification and choose the model order.
    /// </summary>
    public Tensor SingularValues = singularValues;
}
