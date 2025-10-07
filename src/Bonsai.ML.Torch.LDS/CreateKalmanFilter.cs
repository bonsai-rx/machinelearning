using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Creates a Kalman filter model.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Creates a Kalman filter model.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CreateKalmanFilter : IScalarTypeProvider
{
    /// <summary>
    /// A unique name for the Kalman filter model.
    /// </summary>
    [Category("Required Parameters")]
    public string Name { get; set; } = "KalmanFilter";

    private ScalarType _scalarType = ScalarType.Float32;
    /// <inheritdoc/>
    [Description("The data type of the tensor elements.")]
    [TypeConverter(typeof(ScalarTypeConverter))]
    [Category("Required Parameters")]
    public ScalarType Type
    {
        get => _scalarType;
        set
        {
            _scalarType = value;
            ConvertTensorsScalarType(value);
        }
    }

    /// <summary>
    /// The device on which to create the tensor.
    /// </summary>
    [Description("The device on which to create the tensor.")]
    [XmlIgnore]
    [Category("Required Parameters")]
    public Device Device { get; set; }

    private int _numStates = 2;
    /// <summary>
    /// The number of states in the Kalman filter model.
    /// </summary>
    [Category("Required Parameters")]
    public int NumStates
    {
        get => _numStates;
        set => _numStates = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(value), "Number of states must be greater than zero.");
    }

    private int _numObservations = 2;
    /// <summary>
    /// The number of observations in the Kalman filter model.
    /// </summary>
    [Category("Required Parameters")]
    public int NumObservations
    {
        get => _numObservations;
        set => _numObservations = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(value), "Number of observations must be greater than zero.");
    }

    // Tensor properties with XML serialization support
    private Tensor _transitionMatrix;
    /// <summary>
    /// The state transition matrix.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Category("Optional Parameters")]
    public Tensor TransitionMatrix
    {
        get => _transitionMatrix;
        set => _transitionMatrix = value?.to_type(Type);
    }

    /// <summary>
    /// The XML string representation of the transition matrix for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(TransitionMatrix))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string TransitionMatrixXml
    {
        get => TensorConverter.ConvertToString(TransitionMatrix, _scalarType);
        set => TransitionMatrix = TensorConverter.ConvertFromString(value, _scalarType);
    }

    private Tensor _measurementFunction;
    /// <summary>
    /// The measurement function.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Category("Optional Parameters")]
    public Tensor MeasurementFunction
    {
        get => _measurementFunction;
        set => _measurementFunction = value?.to_type(Type);
    }

    /// <summary>
    /// The XML string representation of the measurement function for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(MeasurementFunction))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string MeasurementFunctionXml
    {
        get => TensorConverter.ConvertToString(MeasurementFunction, _scalarType);
        set => MeasurementFunction = TensorConverter.ConvertFromString(value, _scalarType);
    }

    private Tensor _processNoiseVariance;
    /// <summary>
    /// The process noise variance.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Category("Optional Parameters")]
    public Tensor ProcessNoiseVariance
    {
        get => _processNoiseVariance;
        set => _processNoiseVariance = value?.to_type(Type);
    }

    /// <summary>
    /// The XML string representation of the process noise variance for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(ProcessNoiseVariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string ProcessNoiseVarianceXml
    {
        get => TensorConverter.ConvertToString(ProcessNoiseVariance, _scalarType);
        set => ProcessNoiseVariance = TensorConverter.ConvertFromString(value, _scalarType);
    }

    private Tensor _measurementNoiseVariance;
    /// <summary>
    /// The measurement noise variance.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Category("Optional Parameters")]
    public Tensor MeasurementNoiseVariance
    {
        get => _measurementNoiseVariance;
        set => _measurementNoiseVariance = value?.to_type(Type);
    }

    /// <summary>
    /// The XML string representation of the measurement noise variance for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(MeasurementNoiseVariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string MeasurementNoiseVarianceXml
    {
        get => TensorConverter.ConvertToString(MeasurementNoiseVariance, _scalarType);
        set => MeasurementNoiseVariance = TensorConverter.ConvertFromString(value, _scalarType);
    }

    private Tensor _initialMean;
    /// <summary>
    /// The initial mean.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Category("Optional Parameters")]
    public Tensor InitialMean
    {
        get => _initialMean;
        set => _initialMean = value?.to_type(Type);
    }

    /// <summary>
    /// The XML string representation of the initial mean for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(InitialMean))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string InitialMeanXml
    {
        get => TensorConverter.ConvertToString(InitialMean, _scalarType);
        set => InitialMean = TensorConverter.ConvertFromString(value, _scalarType);
    }

    private Tensor _initialCovariance;
    /// <summary>
    /// The initial covariance.
    /// </summary>
    [XmlIgnore]
    [TypeConverter(typeof(TensorConverter))]
    [Category("Optional Parameters")]
    public Tensor InitialCovariance
    {
        get => _initialCovariance;
        set => _initialCovariance = value?.to_type(Type);
    }

    /// <summary>
    /// The XML string representation of the initial covariance for serialization.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(InitialCovariance))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string InitialCovarianceXml
    {
        get => TensorConverter.ConvertToString(InitialCovariance, _scalarType);
        set => InitialCovariance = TensorConverter.ConvertFromString(value, _scalarType);
    }

    private void ConvertTensorsScalarType(ScalarType scalarType)
    {
        _transitionMatrix = _transitionMatrix?.to_type(scalarType);
        _measurementFunction = _measurementFunction?.to_type(scalarType);
        _processNoiseVariance = _processNoiseVariance?.to_type(scalarType);
        _measurementNoiseVariance = _measurementNoiseVariance?.to_type(scalarType);
        _initialMean = _initialMean?.to_type(scalarType);
        _initialCovariance = _initialCovariance?.to_type(scalarType);
    }

    /// <summary>
    /// Creates a Kalman filter model using the properties of this class.
    /// </summary>
    public IObservable<nn.Module> Process()
    {
        return Observable.Using(() => KalmanFilterModelManager.Reserve(
                name: Name,
                numStates: _numStates,
                numObservations: _numObservations,
                transitionMatrix: _transitionMatrix,
                measurementFunction: _measurementFunction,
                initialMean: _initialMean,
                initialCovariance: _initialCovariance,
                processNoiseVariance: _processNoiseVariance,
                measurementNoiseVariance: _measurementNoiseVariance,
                device: Device,
                scalarType: _scalarType
            ), resource => Observable.Return(resource.Model)
                .Concat(Observable.Never(resource.Model))
                .Finally(resource.Dispose)
        );
    }

    /// <summary>
    /// Creates a Kalman filter model using the parameters provided in the input sequence.
    /// </summary>
    public IObservable<nn.Module> Process(IObservable<KalmanFilterParameters> source)
    {
        return source.SelectMany(parameters =>
        {
            return Observable.Using(() => KalmanFilterModelManager.Reserve(
                name: Name,
                parameters: parameters,
                device: Device,
                scalarType: _scalarType
            ), resource => Observable.Return(resource.Model)
                .Concat(Observable.Never(resource.Model))
                .Finally(resource.Dispose)
            );
        });
    }
}
