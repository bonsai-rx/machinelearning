using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using Bonsai.ML.Torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Creates a Kalman filter model.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a Kalman filter model.")]
[WorkflowElementCategory(ElementCategory.Source)]
[TypeConverter(typeof(TensorOperatorConverter))]
public class CreateKalmanFilter : IScalarTypeProvider
{
    /// <inheritdoc/>
    [Description("The data type of the tensor elements.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// The device on which to create the tensor.
    /// </summary>
    [Description("The device on which to create the tensor.")]
    [XmlIgnore]
    public Device Device { get; set; }

    /// <summary>
    /// The number of states in the Kalman filter model.
    /// </summary>
    public int? NumStates { get; set; } = null;

    /// <summary>
    /// The number of observations in the Kalman filter model.
    /// </summary>
    public int? NumObservations { get; set; } = null;
    
    /// <summary>
    /// The state transition matrix.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor TransitionMatrix
    {
        get => _transitionMatrix;
        set => _transitionMatrix = value;
    }

    /// <summary>
    /// The XML string representation of the transition matrix for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(TransitionMatrix))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string TransitionMatrixXml
    {
        get => TensorConverter.ConvertToString(_transitionMatrix, Type);
        set => _transitionMatrix = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The measurement function.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor MeasurementFunction
    {
        get => _measurementFunction;
        set => _measurementFunction = value;
    }

    /// <summary>
    /// The XML string representation of the measurement function for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(MeasurementFunction))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string MeasurementFunctionXml
    {
        get => TensorConverter.ConvertToString(_measurementFunction, Type);
        set => _measurementFunction = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The process noise variance.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor ProcessNoiseVariance
    {
        get => _processNoiseVariance;
        set => _processNoiseVariance = value;
    }

    /// <summary>
    /// The XML string representation of the process noise variance for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(ProcessNoiseVariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ProcessNoiseVarianceXml
    {
        get => TensorConverter.ConvertToString(_processNoiseVariance, Type);
        set => _processNoiseVariance = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The measurement noise variance.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor MeasurementNoiseVariance
    {
        get => _measurementNoiseVariance;
        set => _measurementNoiseVariance = value;
    }

    /// <summary>
    /// The XML string representation of the measurement noise variance for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(MeasurementNoiseVariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string MeasurementNoiseVarianceXml
    {
        get => TensorConverter.ConvertToString(_measurementNoiseVariance, Type);
        set => _measurementNoiseVariance = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The initial mean.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor InitialMean
    {
        get => _initialMean;
        set => _initialMean = value;
    }

    /// <summary>
    /// The XML string representation of the initial mean for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(InitialMean))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string InitialMeanXml
    {
        get => TensorConverter.ConvertToString(_initialMean, Type);
        set => _initialMean = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The initial covariance.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor InitialCovariance
    {
        get => _initialCovariance;
        set => _initialCovariance = value;
    }

    /// <summary>
    /// The XML string representation of the initial covariance for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(InitialCovariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string InitialCovarianceXml
    {
        get => TensorConverter.ConvertToString(_initialCovariance, Type);
        set => _initialCovariance = TensorConverter.ConvertFromString(value, Type);
    }

    private Tensor _transitionMatrix;
    private Tensor _measurementFunction;
    private Tensor _processNoiseVariance;
    private Tensor _measurementNoiseVariance;
    private Tensor _initialMean;
    private Tensor _initialCovariance;

    /// <summary>
    /// Creates a Kalman filter model using the properties of this class.
    /// </summary>
    public IObservable<KalmanFilter> Process()
    {
        return Observable.Return(new KalmanFilter(
            numStates: NumStates,
            numObservations: NumObservations,
            transitionMatrix: TransitionMatrix,
            measurementFunction: MeasurementFunction,
            processNoiseVariance: ProcessNoiseVariance,
            measurementNoiseVariance: MeasurementNoiseVariance,
            initialMean: InitialMean,
            initialCovariance: InitialCovariance,
            device: Device,
            scalarType: Type
        ));
    }

    /// <summary>
    /// Creates a Kalman filter model using the parameters provided in the input sequence.
    /// </summary>
    public IObservable<KalmanFilter> Process(IObservable<KalmanFilterParameters> source)
    {
        return source.Select(parameters =>
        {
            return new KalmanFilter(
                parameters: parameters,
                device: Device,
                scalarType: Type
            );
        });
    }
}
