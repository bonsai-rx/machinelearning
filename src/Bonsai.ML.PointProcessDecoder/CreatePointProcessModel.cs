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

    private StateSpaceType stateSpaceType = StateSpaceType.DiscreteUniform;
    /// <summary>
    /// Gets or sets the type of state space used.
    /// </summary>
    [Category("2. State Space Parameters")]
    [Description("The type of state space used.")]
    public StateSpaceType StateSpaceType
    {
        get
        {
            return stateSpaceType;
        }
        set
        {
            stateSpaceType = value;
        }
    }

    private int stateSpaceDimensions = 1;
    /// <summary>
    /// Gets or sets the number of dimensions in the state space.
    /// </summary>
    [Category("2. State Space Parameters")]
    [Description("The number of dimensions in the state space.")]
    public int StateSpaceDimensions
    {
        get
        {
            return stateSpaceDimensions;
        }
        set
        {
            stateSpaceDimensions = value;
        }
    }

    private double[] minStateSpace = [0];
    /// <summary>
    /// Gets or sets the minimum values of the state space. Must be the same length as the number of state space dimensions.
    /// </summary>
    [Category("2. State Space Parameters")]
    [Description("The minimum values of the state space. Must be the same length as the number of state space dimensions.")]
    public double[] MinStateSpace
    {
        get
        {
            return minStateSpace;
        }
        set
        {
            minStateSpace = value;
        }
    }

    private double[] maxStateSpace = [100];
    /// <summary>
    /// Gets or sets the maximum values of the state space. Must be the same length as the number of state space dimensions.
    /// </summary>
    [Category("2. State Space Parameters")]
    [Description("The maximum values of the state space. Must be the same length as the number of state space dimensions.")]
    public double[] MaxStateSpace
    {
        get
        {
            return maxStateSpace;
        }
        set
        {
            maxStateSpace = value;
        }
    }

    private long[] stepsStateSpace = [50];
    /// <summary>
    /// Gets or sets the number of steps evaluated in the state space. Must be the same length as the number of state space dimensions.
    /// </summary>
    [Category("2. State Space Parameters")]
    [Description("The number of steps evaluated in the state space. Must be the same length as the number of state space dimensions.")]
    public long[] StepsStateSpace
    {
        get
        {
            return stepsStateSpace;
        }
        set
        {
            stepsStateSpace = value;
        }
    }

    private double[] observationBandwidth = [1];
    /// <summary>
    /// Gets or sets the bandwidth of the observation estimation method. Must be the same length as the number of state space dimensions.
    /// </summary>
    [Category("2. State Space Parameters")]
    [Description("The bandwidth of the observation estimation method. Must be the same length as the number of state space dimensions.")]
    public double[] ObservationBandwidth
    {
        get
        {
            return observationBandwidth;
        }
        set
        {
            observationBandwidth = value;
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
    /// Gets or sets the kernel limit.
    /// </summary>
    [Category("3. Encoder Parameters")]
    [Description("The kernel limit.")]
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
    [Category("4. Estimation Parameters")]
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
    /// Gets or sets the distance threshold used to determine the threshold to merge unique clusters into a single compressed cluster.
    /// Only used when the estimation method is set to <see cref="EstimationMethod.KernelCompression"/>.
    /// </summary>
    [Category("4. Estimation Parameters")]
    [Description("The distance threshold used to determine the threshold to merge unique clusters into a single compressed cluster. Only used when the estimation method is set to KernelCompression.")]
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

    private LikelihoodType likelihoodType = LikelihoodType.Poisson;
    /// <summary>
    /// Gets or sets the type of likelihood function used.
    /// </summary>
    [Category("5. Likelihood Parameters")]
    [Description("The type of likelihood function used.")]
    public LikelihoodType LikelihoodType
    {
        get
        {
            return likelihoodType;
        }
        set
        {
            likelihoodType = value;
        }
    }

    private bool ignoreNoSpikes = false;
    /// <summary>
    /// Gets or sets a value indicating whether to ignore contributions from units or channels with no spikes.
    /// </summary>
    [Category("5. Likelihood Parameters")]
    [Description("Indicates whether to ignore contributions from units or channels with no spikes.")]
    public bool IgnoreNoSpikes
    {
        get
        {
            return ignoreNoSpikes;
        }
        set
        {
            ignoreNoSpikes = value;
        }
    }

    private TransitionsType transitionsType = TransitionsType.RandomWalk;
    /// <summary>
    /// Gets or sets the type of transition model used during the decoding process.
    /// </summary>
    [Category("6. Transition Parameters")]
    [Description("The type of transition model used during the decoding process.")]
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
    /// Only used when the transitions type is set to <see cref="TransitionsType.RandomWalk"/> or when the decoder type is set to <see cref="DecoderType.HybridStateSpaceReplayClassifier"/> 
    /// </summary>
    [Category("6. Transition Parameters")]
    [Description("The standard deviation of the random walk transitions model. Only used when the transitions type is set to RandomWalk or when the decoder type is set to HybridStateSpaceReplayClassifier.")]
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
    [Category("7. Decoder Parameters")]
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
    /// Gets or sets the stay probability used in the hybrid state space replay classifier.
    /// Only used when the decoder type is set to <see cref="DecoderType.HybridStateSpaceReplayClassifier"/>.
    /// </summary>
    [Category("7. Decoder Parameters")]
    [Description("The stay probability used in the hybrid state space replay classifier. Only used when the decoder type is set to HybridStateSpaceReplayClassifier.")]
    public double? StayProbability
    {
        get => _stayProbability;
        set
        {
            if (value <= 0 || value >= 1)
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
        return Observable.Using(
            () => PointProcessModelManager.Reserve(
                name: name,
                estimationMethod: estimationMethod,
                transitionsType: transitionsType,
                encoderType: encoderType,
                decoderType: decoderType,
                stateSpaceType: stateSpaceType,
                likelihoodType: likelihoodType,
                minStateSpace: minStateSpace,
                maxStateSpace: maxStateSpace,
                stepsStateSpace: stepsStateSpace,
                observationBandwidth: observationBandwidth,
                stateSpaceDimensions: stateSpaceDimensions,
                markDimensions: markDimensions,
                markChannels: markChannels,
                markBandwidth: markBandwidth,
                ignoreNoSpikes: ignoreNoSpikes,
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