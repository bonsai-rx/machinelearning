using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Linq.Expressions;
using Bonsai.Expressions;
using System.Linq;
using System.Reflection;
using static TorchSharp.torch;
using System.Xml.Serialization;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Creates a PCA model.
/// </summary>
[Combinator]
[ResetCombinator]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeDescriptionProvider(typeof(PcaDescriptionProvider))]
[Description("Creates a PCA model.")]
public class CreatePca : ZeroArgumentExpressionBuilder, INamedElement
{
    /// <inheritdoc/>
    public string Name => ModelType.ToString();

    /// <summary>
    /// The number of principal components to compute.
    /// </summary>
    public int NumComponents { get; set; } = 2;

    /// <summary>
    /// The device on which to create the PCA model.
    /// </summary>
    [XmlIgnore]
    [Description("The device on which to create the PCA model.")]
    public Device? Device { get; set; }

    /// <summary>
    /// The scalar type of the PCA model.
    /// </summary>
    [Description("The scalar type of the PCA model.")]
    public ScalarType? ScalarType { get; set; }

    /// <summary>
    /// The type of PCA model to create.
    /// </summary>
    [RefreshProperties(RefreshProperties.All)]
    [Description("The type of PCA model to create.")]
    public PcaModelType ModelType { get; set; } = PcaModelType.Pca;

    /// <summary>
    /// The initial variance for probabilistic PCA models.
    /// </summary>
    [Description("The initial variance for probabilistic PCA models.")]
    public double InitialVariance { get; set; } = 1.0;

    /// <summary>
    /// The number of iterations for fitting probabilistic PCA models.
    /// </summary>
    [Description("The number of iterations for fitting probabilistic PCA models.")]
    public int Iterations { get; set; } = 100;

    /// <summary>
    /// The tolerance for convergence in probabilistic PCA models.
    /// </summary>
    [Description("The tolerance for convergence in probabilistic PCA models.")]
    public double Tolerance { get; set; } = 1e-5;

    /// <summary>
    /// The constant learning rate parameter for the online probabilistic PCA model.
    /// </summary>
    [Description("The constant learning rate parameter for the online probabilistic PCA model. Only one of Rho or Kappa must be specified.")]
    public double? Rho { get; set; } = 0.1;

    /// <summary>
    /// The forgetting factor for the online probabilistic PCA model.
    /// </summary>
    [Description("The forgetting factor for the online probabilistic PCA model. Only one of Rho or Kappa must be specified.")]
    public double? Kappa { get; set; } = 0.9;

    /// <summary>
    /// The time offset for the online probabilistic PCA model.
    /// </summary>
    [Description("The time offset for the online probabilistic PCA model. If null, decaying learning rate starts from the first sample.")]
    public int? TimeOffset { get; set; } = null;

    /// <summary>
    /// The period for reorthogonalizing the components in the online probabilistic PCA model.
    /// </summary>
    [Description("The period for reorthogonalizing the components in the online probabilistic PCA model. If null, reorthogonalization is not performed.")]
    public int? ReorthogonalizePeriod { get; set; } = null;

    /// <summary>
    /// The random number generator used for initializing probabilistic PCA models.
    /// </summary>
    [XmlIgnore]
    public Generator? Generator { get; set; } = null;

    internal IEnumerable<string> GetModelProperties()
    {
        yield return nameof(NumComponents);
        yield return nameof(Device);
        yield return nameof(ScalarType);
        yield return nameof(ModelType);

        if (ModelType == PcaModelType.ProbabilisticPca)
        {
            yield return nameof(InitialVariance);
            yield return nameof(Iterations);
            yield return nameof(Tolerance);
            yield return nameof(Generator);
        }

        if (ModelType == PcaModelType.OnlineProbabilisticPca)
        {
            yield return nameof(InitialVariance);
            yield return nameof(Rho);
            yield return nameof(Kappa);
            yield return nameof(TimeOffset);
            yield return nameof(ReorthogonalizePeriod);
            yield return nameof(Generator);
        }
    }

    private static PcaBaseModel CreateModel(CreatePca pcaBuilder)
    {
        return pcaBuilder.ModelType switch
        {
            PcaModelType.Pca => new Pca(
                numComponents: pcaBuilder.NumComponents,
                device: pcaBuilder.Device,
                scalarType: pcaBuilder.ScalarType),
            PcaModelType.ProbabilisticPca => new ProbabilisticPca(
                numComponents: pcaBuilder.NumComponents,
                device: pcaBuilder.Device,
                scalarType: pcaBuilder.ScalarType,
                initialVariance: pcaBuilder.InitialVariance,
                generator: pcaBuilder.Generator,
                iterations: pcaBuilder.Iterations,
                tolerance: pcaBuilder.Tolerance),
            PcaModelType.OnlineProbabilisticPca => new OnlineProbabilisticPca(
                numComponents: pcaBuilder.NumComponents,
                device: pcaBuilder.Device,
                scalarType: pcaBuilder.ScalarType,
                initialVariance: pcaBuilder.InitialVariance,
                generator: pcaBuilder.Generator,
                rho: pcaBuilder.Rho,
                kappa: pcaBuilder.Kappa,
                timeOffset: pcaBuilder.TimeOffset,
                reorthogonalizePeriod: pcaBuilder.ReorthogonalizePeriod),
            _ => throw new NotSupportedException($"Model type {pcaBuilder.ModelType} is not supported."),
        };
    }

    private static Type GetModelType(PcaModelType modelType)
    {
        return modelType switch
        {
            PcaModelType.Pca => typeof(Pca),
            PcaModelType.ProbabilisticPca => typeof(ProbabilisticPca),
            PcaModelType.OnlineProbabilisticPca => typeof(OnlineProbabilisticPca),
            _ => throw new NotSupportedException($"Model type {modelType} is not supported."),
        };
    }

    /// <inheritdoc/>
    public override Expression Build(IEnumerable<Expression> arguments)
    {
        var processMethod = GetType().GetMethod(
            nameof(Process),
            BindingFlags.NonPublic | BindingFlags.Static);
        processMethod = processMethod.MakeGenericMethod(GetModelType(ModelType));
        return Expression.Call(processMethod, Expression.Constant(this));
    }

    private static IObservable<T> Process<T>(CreatePca instance) where T : PcaBaseModel
    {
        return Observable.Return((T)CreateModel(instance));
    }
}
