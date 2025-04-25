using System;
using System.ComponentModel;
using System.Xml.Serialization;
using System.Linq;
using System.Reactive.Linq;

using static TorchSharp.torch;

using PointProcessDecoder.Core;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.Decoder;
using PointProcessDecoder.Core.StateSpace;
using PointProcessDecoder.Core.Likelihood;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Creates a new neural decoding model based on point processes using Bayesian state space models.
/// </summary>
[Combinator]
[WorkflowElementCategory(ElementCategory.Source)]
[Description("Creates a new neural decoding model based on point processes using Bayesian state space models.")]
public class CreatePointProcessModel : IManagedPointProcessModelNode
{
    private string name = "PointProcessModel";
    /// <summary>
    /// Gets or sets the name of the point process model.
    /// </summary>
    [Category("1. Model Parameters")]
    [Description("The name of the point process model.")]
    public string Name
    {
        get
        {
            return name;
        }
        set
        {
            name = value;
        }
    }

    Device? device = null;
    /// <summary>
    /// Gets or sets the device used to run the neural decoding model.
    /// </summary>
    [XmlIgnore]
    [Category("1. Model Parameters")]
    [Description("The device used to run the neural decoding model.")]
    public Device? Device
    {
        get
        {
            return device;
        }
        set
        {
            device = value;
        }
    }

    ScalarType? scalarType = null;
    /// <summary>
    /// Gets or sets the scalar type used to run the neural decoding model.
    /// </summary>
    [Category("1. Model Parameters")]
    [Description("The scalar type used to run the neural decoding model.")]
    public ScalarType? ScalarType
    {
        get
        {
            return scalarType;
        }
        set
        {
            scalarType = value;
        }
    }

    private readonly StateSpaceType stateSpaceType = StateSpaceType.DiscreteUniform;

    private int covariateDimensions = 1;
    /// <summary>
    /// Gets or sets the number of dimensions of the covariate data.
    /// </summary>
    [Category("2. Covariate Parameters")]
    [Description("The number of dimensions of the covariate.")]
    public int Dimensions
    {
        get
        {
            return covariateDimensions;
        }
        set
        {
            covariateDimensions = value;
        }
    }

    private double[] minCovariateRange = [0];
    /// <summary>
    /// Gets or sets the minimum values of the covariate range. Must be the same length as the number of covariate dimensions.
    /// </summary>
    [Category("2. Covariate Parameters")]
    [Description("The minimum values of the covariate range. Must be the same length as the number of covariate dimensions.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public double[] MinRange
    {
        get
        {
            return minCovariateRange;
        }
        set
        {
            minCovariateRange = value;
        }
    }

    private double[] maxCovariateRange = [100];
    /// <summary>
    /// Gets or sets the maximum values of the covariate range. Must be the same length as the number of covariate dimensions.
    /// </summary>
    [Category("2. Covariate Parameters")]
    [Description("The maximum values of the covariate range. Must be the same length as the number of covariate dimensions.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public double[] MaxRange
    {
        get
        {
            return maxCovariateRange;
        }
        set
        {
            maxCovariateRange = value;
        }
    }

    private long[] stepsCovariateRange = [50];
    /// <summary>
    /// Gets or sets the number of steps evaluated in the covariate range. Must be the same length as the number of covariate dimensions.
    /// </summary>
    [Category("2. Covariate Parameters")]
    [Description("The number of steps evaluated in the covariate range. Must be the same length as the number of covariate dimensions.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] Steps
    {
        get
        {
            return stepsCovariateRange;
        }
        set
        {
            stepsCovariateRange = value;
        }
    }

    private double[] covariateBandwidth = [1];
    /// <summary>
    /// Gets or sets the kernel bandwidth used to estimate the probability density over the covariate dimensions. Must be the same length as the covariate dimensions.
    /// </summary>
    [Category("2. Covariate Parameters")]
    [Description("The kernel bandwidth used to estimate the probability density over the covariate dimensions. Must be the same length as the covariate dimensions.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public double[] Bandwidth
    {
        get
        {
            return covariateBandwidth;
        }
        set
        {
            covariateBandwidth = value;
        }
    }

    private EncoderType encoderType = EncoderType.SortedSpikes;
    /// <summary>
    /// Gets or sets the type of encoder used.
    /// </summary>
    [Category("3. Encoder Parameters")]
    [Description("The type of encoder used.")]
    public EncoderType EncoderType
    {
        get
        {
            return encoderType;
        }
        set
        {
            encoderType = value;
        }
    }

    private int? kernelLimit = null;
    /// <summary>
    /// Gets or sets the maximum number of kernels maintained in memory for each probability density estimation made by the encoder. In the case of sorted spikes, there is an estimate for the full covariate distribution and an estimate for each unit. In the case of clusterless marks, there is an estimate for the full covariate distribution, and 2 estimates for each mark channel (1 for the overall distribution of marks and 1 for the joint distribution of covariates and marks).
    /// </summary>
    [Category("3. Encoder Parameters")]
    [Description("The maximum number of kernels maintained in memory for each probability density estimation made by the encoder. In the case of sorted spikes, there is an estimate for the full covariate distribution and an estimate for each unit. In the case of clusterless marks, there is an estimate for the full covariate distribution, and 2 estimates for each mark channel (1 for the overall distribution of marks and 1 for the joint distribution of covariates and marks).")]
    public int? KernelLimit
    {
        get
        {
            return kernelLimit;
        }
        set
        {
            kernelLimit = value;
        }
    }

    private int? nUnits = null;
    /// <summary>
    /// Gets or sets the number of sorted spiking units.
    /// Only used when the encoder type is set to <see cref="EncoderType.SortedSpikes"/>.
    /// </summary>
    [Category("3. Encoder Parameters")]
    [Description("The number of sorted spiking units. Only used when the encoder type is set to SortedSpikeEncoder.")]
    public int? NUnits
    {
        get
        {
            return nUnits;
        }
        set
        {
            nUnits = value;
        }
    }

    private int? markDimensions = null;
    /// <summary>
    /// Gets or sets the number of dimensions or features associated with each mark.
    /// Only used when the encoder type is set to <see cref="EncoderType.ClusterlessMarks"/>.
    /// </summary>
    [Category("3. Encoder Parameters")]
    [Description("The number of dimensions or features associated with each mark. Only used when the encoder type is set to ClusterlessMarkEncoder.")]
    public int? MarkDimensions
    {
        get
        {
            return markDimensions;
        }
        set
        {
            markDimensions = value;
        }
    }

    private int? markChannels = null;
    /// <summary>
    /// Gets or sets the number of mark recording channels.
    /// Only used when the encoder type is set to <see cref="EncoderType.ClusterlessMarks"/>.
    /// </summary>
    [Category("3. Encoder Parameters")]
    [Description("The number of mark recording channels. Only used when the encoder type is set to ClusterlessMarkEncoder.")]
    public int? MarkChannels
    {
        get
        {
            return markChannels;
        }
        set
        {
            markChannels = value;
        }
    }

    private double[]? markBandwidth = null;
    /// <summary>
    /// Gets or sets the bandwidth of the mark estimation method.
    /// Must be the same length as the number of mark dimensions.
    /// Only used when the encoder type is set to <see cref="EncoderType.ClusterlessMarks"/>.
    /// </summary>
    [Category("3. Encoder Parameters")]
    [Description("The bandwidth of the mark estimation method. Must be the same length as the number of mark dimensions. Only used when the encoder type is set to ClusterlessMarkEncoder.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public double[]? MarkBandwidth
    {
        get
        {
            return markBandwidth;
        }
        set
        {
            markBandwidth = value;
        }
    }

    private EstimationMethod estimationMethod = EstimationMethod.KernelDensity;
    /// <summary>
    /// Gets or sets the estimation method used during the encoding process.
    /// </summary>
    [Category("3. Encoder Parameters")]
    [Description("The estimation method used during the encoding process.")]
    public EstimationMethod EstimationMethod
    {
        get
        {
            return estimationMethod;
        }
        set
        {
            estimationMethod = value;
        }
    }

    private double? distanceThreshold = null;
    /// <summary>
    /// Gets or sets the distance threshold used to determine if a new data point is merged into an existing kernel or if a new kernel gets created.
    /// Only used when the estimation method is set to <see cref="EstimationMethod.KernelCompression"/>.
    /// </summary>
    [Category("3. Encoder Parameters")]
    [Description("The distance threshold used to determine if a new data point is merged into an existing kernel or if a new kernel gets created. Only used when the estimation method is set to KernelCompression.")]
    public double? DistanceThreshold
    {
        get
        {
            return distanceThreshold;
        }
        set
        {
            distanceThreshold = value;
        }
    }

    private TransitionsType transitionsType = TransitionsType.RandomWalk;
    /// <summary>
    /// Gets or sets the type of transition model used during the decoding process.
    /// Only used when the decoder type is set to <see cref="DecoderType.StateSpaceDecoder"/>.
    /// </summary>
    [Category("4. Decoder Parameters")]
    [Description("The type of transition model used during the decoding process. Only used when the decoder type is set to StateSpaceDecoder.")]
    public TransitionsType TransitionsType
    {
        get
        {
            return transitionsType;
        }
        set
        {
            transitionsType = value;
        }
    }

    private double? sigmaRandomWalk = null;
    /// <summary>
    /// Gets or sets the standard deviation of the random walk transitions model.
    /// Only used when the transitions type is set to <see cref="TransitionsType.RandomWalk"/> or when the decoder type is set to <see cref="DecoderType.HybridStateSpaceClassifier"/> 
    /// </summary>
    [Category("4. Decoder Parameters")]
    [Description("The standard deviation of the random walk transitions model. Only used when the transitions type is set to RandomWalk or when the decoder type is set to HybridStateSpaceClassifier.")]
    public double? SigmaRandomWalk
    {
        get
        {
            return sigmaRandomWalk;
        }
        set
        {
            sigmaRandomWalk = value;
        }
    }

    private DecoderType decoderType = DecoderType.StateSpaceDecoder;
    /// <summary>
    /// Gets or sets the type of decoder used.
    /// </summary>
    [Category("4. Decoder Parameters")]
    [Description("The type of decoder used.")]
    public DecoderType DecoderType
    {
        get
        {
            return decoderType;
        }
        set
        {
            decoderType = value;
        }
    }

    private double? _stayProbability = null;
    /// <summary>
    /// Gets or sets the stay probability used in the discrete transition matrix.
    /// Only used when the decoder type is set to <see cref="DecoderType.HybridStateSpaceClassifier"/>.
    /// </summary>
    [Category("4. Decoder Parameters")]
    [Description("The stay probability used in the discrete transition matrix. Only used when the decoder type is set to HybridStateSpaceClassifier.")]
    public double? StayProbability
    {
        get => _stayProbability;
        set
        {
            if (value < 0 || value > 1)
            {
                throw new ArgumentOutOfRangeException(nameof(StayProbability), "The stay probability must be greater than zero and less than one.");
            }

            _stayProbability = value;
        }
    }

    /// <summary>
    /// Creates a new neural decoding model based on point processes using Bayesian state space models.
    /// </summary>
    /// <returns></returns>
    public IObservable<PointProcessModel> Process()
    {
        var likelihoodType = encoderType switch
        {
            EncoderType.SortedSpikes => LikelihoodType.Poisson,
            EncoderType.ClusterlessMarks => LikelihoodType.Clusterless,
            _ => throw new NotSupportedException($"The encoder type {encoderType} is not supported.")
        };


        return Observable.Using(
            () => PointProcessModelManager.Reserve(
                name: name,
                estimationMethod: estimationMethod,
                transitionsType: transitionsType,
                encoderType: encoderType,
                decoderType: decoderType,
                stateSpaceType: stateSpaceType,
                likelihoodType: likelihoodType,
                minStateSpace: minCovariateRange,
                maxStateSpace: maxCovariateRange,
                stepsStateSpace: stepsCovariateRange,
                observationBandwidth: covariateBandwidth,
                stateSpaceDimensions: covariateDimensions,
                markDimensions: markDimensions,
                markChannels: markChannels,
                markBandwidth: markBandwidth,
                nUnits: nUnits,
                distanceThreshold: distanceThreshold,
                sigmaRandomWalk: sigmaRandomWalk,
                kernelLimit: kernelLimit,
                stayProbability: _stayProbability,
                device: device,
                scalarType: scalarType
            ), resource => Observable.Return(resource.Model)
                .Concat(Observable.Never(resource.Model))
                .Finally(resource.Dispose));
    }
}