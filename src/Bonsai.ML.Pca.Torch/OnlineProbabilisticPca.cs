using System;
using System.ComponentModel;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.linalg;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// An online probabilistic PCA model which uses a stochastic online EM algorithm.
/// </summary>
/// <remarks>
/// The first batch of data initializes the mean. The running statistics start from their
/// expected values under the initial model, so the initial variance should be of the same order
/// as the variance of the data. The components span the principal subspace, and are aligned to
/// the principal axes only when reorthogonalization is enabled.
/// </remarks>
public class OnlineProbabilisticPca : PcaBaseModel
{
    private Tensor _identityComponents = empty(0);
    private Tensor _Cxz = empty(0); // E[xz^T]
    private Tensor _mz = empty(0); // E[z]
    private Tensor _Czz = empty(0); // E[zz^T]
    private Tensor _sxx = empty(0); // E[||x||^2]
    private long _stepCount = 0;

    /// <summary>
    /// Gets the mean of the fitted data.
    /// </summary>
    [XmlIgnore]
    public Tensor Mean { get; private set; } = empty(0);

    /// <summary>
    /// Gets the variance of the isotropic Gaussian noise model.
    /// </summary>
    [XmlIgnore]
    public double Variance { get; private set; } = 0.0;

    /// <summary>
    /// Gets or sets the constant learning rate parameter.
    /// </summary>
    /// <remarks>
    /// Rho must be in the range (0, 1). Only one of Rho or Kappa should be specified.
    /// </remarks>
    [Category("ModelParameters")]
    [Description("The constant learning rate parameter. Only one of Rho or Kappa should be specified.")]
    public double? Rho { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the exponent in the learning rate schedule.
    /// </summary>
    /// <remarks>
    /// Kappa must be in the range (0.5, 1]. The learning rate of update t, counting from one, is
    /// (t + 1 + SampleOffset)^(-Kappa). Only one of Rho or Kappa should be specified.
    /// </remarks>
    [Category("ModelParameters")]
    [Description("The exponent in the learning rate schedule. Only one of Rho or Kappa should be specified.")]
    public double? Kappa { get; set; }

    /// <summary>
    /// Gets or sets the initial variance of the isotropic Gaussian noise model.
    /// </summary>
    [Category("ModelParameters")]
    [Description("The initial variance of the isotropic Gaussian noise model.")]
    public double InitialVariance { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the period for reorthogonalizing the principal components.
    /// </summary>
    /// <remarks>
    /// Represented as the number of update steps between reorthogonalization operations.
    /// If not specified, reorthogonalization is not performed.
    /// </remarks>
    [Category("ModelParameters")]
    [Description("The number of update steps between reorthogonalization operations. If not specified, reorthogonalization is not performed.")]
    public int? ReorthogonalizePeriod { get; set; }

    /// <summary>
    /// Gets or sets the sample offset used in the learning rate schedule when Kappa is specified.
    /// </summary>
    [Category("ModelParameters")]
    [Description("The sample offset used in the learning rate schedule when Kappa is specified. If not specified, the decaying learning rate starts from the first sample.")]
    public int? SampleOffset { get; set; }

    /// <summary>
    /// Gets or sets the random number generator used for initializing the model.
    /// </summary>
    [XmlIgnore]
    [Description("The random number generator used for initializing the model.")]
    public Generator? Generator { get; set; }

    /// <inheritdoc/>
    protected override bool IsIncremental => true;

    /// <inheritdoc/>
    protected override void ValidateParameters()
    {
        base.ValidateParameters();

        if (InitialVariance <= 0)
        {
            throw new InvalidOperationException("Initial variance must be greater than zero.");
        }

        if (Kappa.HasValue && Rho.HasValue)
        {
            throw new InvalidOperationException("Only one of Rho or Kappa should be specified, not both.");
        }

        if (!Kappa.HasValue && !Rho.HasValue)
        {
            throw new InvalidOperationException("Either Rho or Kappa must be specified.");
        }

        if (Rho.HasValue && (Rho.Value <= 0 || Rho.Value >= 1))
        {
            throw new InvalidOperationException("Rho must be in the range (0, 1).");
        }

        if (Kappa.HasValue && (Kappa.Value <= 0.5 || Kappa.Value > 1))
        {
            throw new InvalidOperationException("Kappa must be in the range (0.5, 1].");
        }

        if (SampleOffset < 0)
        {
            throw new InvalidOperationException("Sample offset must be greater than or equal to zero.");
        }

        if (ReorthogonalizePeriod <= 0)
        {
            throw new InvalidOperationException("Reorthogonalize period must be greater than zero.");
        }
    }

    private double GetUpdateRate()
    {
        return Rho ?? Math.Pow(_stepCount + 1 + (SampleOffset ?? 0), -Kappa!.Value);
    }

    private void Initialize(Tensor data)
    {
        var numFeatures = data.size(1);
        var type = data.dtype;
        var device = data.device;

        var basis = qr(randn(numFeatures, NumComponents, dtype: type, device: device, generator: Generator), mode: QRMode.Reduced).Q;
        var weights = basis * Math.Sqrt(InitialVariance);
        var mean = data.mean([0], keepdim: true);

        Variance = InitialVariance;
        Components = weights.DetachFromDisposeScope();
        Mean = mean.DetachFromDisposeScope();

        // Expected statistics under the initial model
        _identityComponents.Dispose();
        _identityComponents = eye(NumComponents, dtype: type, device: device).DetachFromDisposeScope(); // q x q

        _Cxz.Dispose();
        _Cxz = weights.clone().DetachFromDisposeScope(); // d x q

        _mz.Dispose();
        _mz = zeros(NumComponents, dtype: type, device: device).DetachFromDisposeScope(); // q

        _Czz.Dispose();
        _Czz = eye(NumComponents, dtype: type, device: device).DetachFromDisposeScope(); // q x q

        _sxx.Dispose();
        _sxx = (mean.pow(2).sum() + weights.pow(2).sum() + numFeatures * Variance).DetachFromDisposeScope(); // scalar
    }

    /// <inheritdoc/>
    protected override void FitCore(Tensor data)
    {
        if (_stepCount == 0)
            Initialize(data);

        _stepCount++;
        var rho = GetUpdateRate();
        var numSamples = data.size(0);
        var numFeatures = data.size(1);

        // Center data using current mean
        var dataCentered = data - Mean;

        // E-step
        var M = Components.T.matmul(Components) + _identityComponents * Variance;
        var MInv = Utils.SolveSpd(M, _identityComponents);
        var projection = dataCentered.matmul(Components);
        var EzT = Utils.SolveSpd(M, projection.T);
        var Ez = EzT.T;

        // Batch statistics
        var mxBatch = data.mean([0], keepdim: true);
        var sxxBatch = data.pow(2).sum(dim: 1).mean();
        var CxzBatch = data.T.matmul(Ez) / numSamples;
        var mzBatch = Ez.mean([0]);
        var CzzBatch = EzT.matmul(Ez) / numSamples + Variance * MInv;

        // Update statistics
        var rhoFactor = 1 - rho;
        var mean = rhoFactor * Mean + rho * mxBatch;
        var Cxz = rhoFactor * _Cxz + rho * CxzBatch;
        var mz = rhoFactor * _mz + rho * mzBatch;
        var sxx = rhoFactor * _sxx + rho * sxxBatch;
        var Czz = rhoFactor * _Czz + rho * CzzBatch;

        // M-step
        var Sxz = Cxz - mean.T * mz;
        var weightsUpdated = Utils.SolveSpd(Czz, Sxz.T).T;

        if (ReorthogonalizePeriod.HasValue &&
            _stepCount % ReorthogonalizePeriod.Value == 0)
        {
            var (U, S, Vh) = svd(weightsUpdated, fullMatrices: false);
            var R = Vh.T;
            weightsUpdated = U.matmul(diag(S));
            Cxz = Cxz.matmul(R);
            Czz = R.T.matmul(Czz).matmul(R);
            mz = R.T.matmul(mz);
        }

        // Reorder components based on the strength of the components
        var strength = sum(weightsUpdated * weightsUpdated, dim: 0);
        var indices = argsort(strength, descending: true);
        var weights = weightsUpdated.index_select(1, indices);
        Cxz = Cxz.index_select(1, indices);
        mz = mz.index_select(0, indices);
        Czz = Czz.index_select(0, indices).index_select(1, indices);

        // Centered statistics
        Sxz = Cxz - mean.T * mz;
        var Sxx = sxx - mean.pow(2).sum();

        // Update variance
        Variance = ((Sxx - 2 * trace(weights.T.matmul(Sxz)) + trace(weights.T.matmul(weights).matmul(Czz))) / (double)numFeatures)
            .clamp_min(0.0)
            .to_type(ScalarType.Float64)
            .item<double>();

        Components = weights.DetachFromDisposeScope();
        Mean = mean.DetachFromDisposeScope();

        _Cxz.Dispose();
        _Cxz = Cxz.DetachFromDisposeScope();

        _mz.Dispose();
        _mz = mz.DetachFromDisposeScope();

        _Czz.Dispose();
        _Czz = Czz.DetachFromDisposeScope();

        _sxx.Dispose();
        _sxx = sxx.DetachFromDisposeScope();
    }

    /// <inheritdoc/>
    protected override Tensor TransformCore(Tensor data)
    {
        return Utils.PosteriorMean(data - Mean, Components, Variance);
    }

    /// <inheritdoc/>
    protected override Tensor ReconstructCore(Tensor data)
    {
        return Utils.Project(data, Components, Variance) + Mean;
    }

    /// <inheritdoc/>
    public override void Dispose()
    {
        base.Dispose();

        Mean.Dispose();
        Mean = empty(0);
        Variance = 0.0;

        _stepCount = 0;

        _identityComponents.Dispose();
        _identityComponents = empty(0);

        _Cxz.Dispose();
        _Cxz = empty(0);

        _mz.Dispose();
        _mz = empty(0);

        _Czz.Dispose();
        _Czz = empty(0);

        _sxx.Dispose();
        _sxx = empty(0);
    }
}
