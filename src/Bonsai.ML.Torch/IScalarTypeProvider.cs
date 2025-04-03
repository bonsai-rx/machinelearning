using static TorchSharp.torch;

namespace Bonsai.ML.Torch;

/// <summary>
/// An interface that defines a class which provides scalar type information.
/// </summary>
public interface IScalarTypeProvider
{
    /// <summary>
    /// The data type of the tensor elements.
    /// </summary>
    public ScalarType Type { get; }
}
