using System;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Implements Online PCA using the Generalized Hebbian Algorithm (GHA).
/// </summary>
/// <param name="numComponents"></param>
/// <param name="learningRate"></param>
/// <param name="device"></param>
/// <param name="scalarType"></param>
/// <param name="generator"></param>
public class OnlinePcaGha(
    int numComponents,
    double learningRate = 0.1,
    Device? device = null,
    ScalarType? scalarType = null,
    Generator? generator = null
) : PcaBaseModel(numComponents, device, scalarType)
{

    private Tensor _mean = empty(0);
    private int _sampleCount = 0;

    /// <summary>
    /// Gets or sets the learning rate for the GHA algorithm.
    /// </summary>
    public double LearningRate { get; set; } = learningRate;

    /// <inheritdoc/>
    public override Tensor Components { get; protected set; } = empty(0);

    /// <summary>
    /// Gets the random number generator used for initializing the model.
    /// </summary>
    public Generator Generator { get; private set; } = generator ?? manual_seed(0);

    /// <inheritdoc/>
    public override void Fit(Tensor data)
    {
        // throw new NotImplementedException();
        if (data.NumberOfElements == 0 || data.dim() != 2)
        {
            throw new ArgumentException("Input data must be a 2D tensor.");
        }

        var q = NumComponents;

        // Data is shaped (number of samples x number of features)
        var n = data.shape[0];
        var d = data.shape[1];

        if (q > d)
        {
            throw new ArgumentException("Number of components cannot be greater than the number of features.", nameof(data));
        }

        // Initialize components randomly
        if (Components.numel() == 0)
            Components = linalg.qr(randn([d, q], ScalarType, Device), mode: linalg.QRMode.Reduced).Q;

        if (_mean.numel() == 0)
            _mean = data.mean([0], keepdim: true);
        else
            _mean.mul_(_sampleCount / (double)(_sampleCount + n)).add_(data.mean([0], keepdim: true), alpha: n / (double)(_sampleCount + n));

        _sampleCount += (int)n;
        var dataCentered = data - _mean;

        var Y = dataCentered.matmul(Components); // n x q
        var hebbianTerm = dataCentered.T.matmul(Y); // d x q
        var crossTerm = Y.T.matmul(Y); // q x q
        var lowerTriangular = crossTerm.tril(0); // q x q
        var correlation = lowerTriangular.matmul(Components.T); // q x d

        // Update components
        Components.add_(hebbianTerm - correlation.T, alpha: LearningRate);
    }

    /// <inheritdoc/>
    public override Tensor Transform(Tensor data)
    {
        if (data.NumberOfElements == 0 || data.dim() != 2)
        {
            throw new ArgumentException("Input data must be a 2D tensor.");
        }
        var dataCentered = data - _mean;
        return dataCentered.matmul(Components.T);
    }

    /// <inheritdoc/>
    public override Tensor Reconstruct(Tensor data)
    {
        var transformed = Transform(data);
        return transformed.matmul(Components.T) + _mean;
    }
}
