namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Specifies the type of PCA model.
/// </summary>
public enum PcaModelType
{
    /// <summary>
    /// Standard PCA model.
    /// </summary>
    Pca,

    /// <summary>
    /// Probabilistic PCA model.
    /// </summary>
    ProbabilisticPca,

    /// <summary>
    /// Online Probabilistic PCA model.
    /// </summary>
    OnlineProbabilisticPca
}
