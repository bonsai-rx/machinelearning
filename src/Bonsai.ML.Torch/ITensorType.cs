using static TorchSharp.torch;

namespace Bonsai.ML.Torch;

/// <summary>
/// An interface that defines a class which contains tensor type information.
/// </summary>
public interface ITensorType
{
    /// <summary>
    /// The data type of the tensor elements.
    /// </summary>
    public ScalarType Type { get; set; }
}
