using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Provides an abstract base class for PCA models.
/// </summary>
public abstract class PcaBaseModel : IPcaBaseModel
{
    /// <inheritdoc/>
    public abstract Tensor Components { get; protected set; }

    /// <inheritdoc/>
    public int NumComponents { get; private set; }

    /// <inheritdoc/>
    public Device Device { get; private set; }

    /// <inheritdoc/>
    public ScalarType ScalarType { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="PcaBaseModel"/> class.
    /// </summary>
    /// <param name="numComponents"></param>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    /// <exception cref="ArgumentException"></exception>
    public PcaBaseModel(int numComponents,
        Device? device = null,
        ScalarType? scalarType = null)
    {
        if (numComponents <= 0)
        {
            throw new ArgumentException("Number of components must be greater than zero.", nameof(numComponents));
        }

        NumComponents = numComponents;
        Device = device ?? CPU;
        ScalarType = scalarType ?? ScalarType.Float32;
    }

    /// <inheritdoc/>
    public abstract void Fit(Tensor data);

    /// <inheritdoc/>
    public abstract Tensor Transform(Tensor data);

    /// <inheritdoc/>
    public virtual Tensor FitAndTransform(Tensor data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc/>
    public abstract Tensor Reconstruct(Tensor data);
}
