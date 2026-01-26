namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Defines an interface for PCA model providers.
/// </summary>
public interface IPcaModelProvider
{
    /// <summary>
    /// Gets or sets the PCA model.
    /// </summary>
    public IPcaBaseModel? Model { get; set; }
}
