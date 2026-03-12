using PointProcessDecoder.Core.Decoder;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Represents a packaged data frame containing the output of a point process decoder model.
/// </summary>
/// <param name="decoderData"></param>
/// <param name="name"></param>
public readonly struct DecoderDataFrame(
    DecoderData decoderData,
    string name) : IPointProcessModelReference
{
    /// <summary>
    /// The packaged decoder data.
    /// </summary>
    public DecoderData DecoderData => decoderData;

    /// <summary>
    /// The name of the point process model.
    /// </summary>
    public string Name => name;
}