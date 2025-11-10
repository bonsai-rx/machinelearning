using PointProcessDecoder.Core.Decoder;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Represents a packaged data frame containing the output of a point process classifier model.
/// </summary>
/// <param name="classifierData"></param>
/// <param name="name"></param>
public readonly struct ClassifierDataFrame(
    ClassifierData classifierData,
    string name) : IPointProcessModelReference
{
    /// <summary>
    /// The packaged classifier data.
    /// </summary>
    public ClassifierData ClassifierData => classifierData;

    /// <summary>
    /// The name of the point process model.
    /// </summary>
    public string Name => name;
}