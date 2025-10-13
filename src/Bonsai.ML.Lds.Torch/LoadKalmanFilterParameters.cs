using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Xml.Serialization;
using Bonsai.ML.Torch;
using System.IO;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Loads the parameters of a Kalman filter model.
/// </summary>
[Combinator]
[Description("Loads the parameters of a Kalman filter model.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LoadKalmanFilterParameters
{
    /// <summary>
    /// Reads the path to a .bin file containing the transition matrix.
    /// </summary>
    [Description("Reads the path to a .bin file containing the transition matrix.")]
    [FileNameFilter("Binary Data (*.bin)|*.bin|All Files|*.*")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string TransitionMatrixFilePath { get; set; } = "transition_matrix.bin";

    /// <summary>
    /// Reads the path to a .bin file containing the measurement function.
    /// </summary>
    [Description("Reads the path to a .bin file containing the measurement function.")]
    [FileNameFilter("Binary Data (*.bin)|*.bin|All Files|*.*")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string MeasurementFunctionFilePath { get; set; } = "measurement_function.bin";

    /// <summary>
    /// Reads the path to a .bin file containing the process noise covariance.
    /// </summary>
    [Description("Reads the path to a .bin file containing the process noise covariance.")]
    [FileNameFilter("Binary Data (*.bin)|*.bin|All Files|*.*")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string ProcessNoiseCovarianceFilePath { get; set; } = "process_noise_covariance.bin";

    /// <summary>
    /// Reads the path to a .bin file containing the measurement noise covariance.
    /// </summary>
    [Description("Reads the path to a .bin file containing the measurement noise covariance.")]
    [FileNameFilter("Binary Data (*.bin)|*.bin|All Files|*.*")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string MeasurementNoiseCovarianceFilePath { get; set; } = "measurement_noise_covariance.bin";

    /// <summary>
    /// Reads the path to a .bin file containing the initial mean.
    /// </summary>
    [Description("Reads the path to a .bin file containing the initial mean.")]
    [FileNameFilter("Binary Data (*.bin)|*.bin|All Files|*.*")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string InitialMeanFilePath { get; set; } = "initial_mean.bin";

    /// <summary>
    /// Reads the path to a .bin file containing the initial covariance.
    /// </summary>
    [Description("Reads the path to a .bin file containing the initial covariance.")]
    [FileNameFilter("Binary Data (*.bin)|*.bin|All Files|*.*")]
    [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string InitialCovarianceFilePath { get; set; } = "initial_covariance.bin";

    /// <summary>
    /// Gets or sets the data type of the tensors.
    /// </summary>
    [Description("Gets or sets the data type of the tensors.")]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Gets or sets the device to use for tensor operations.
    /// </summary>
    [XmlIgnore]
    [Description("Gets or sets the device to use for tensor operations.")]
    public Device Device { get; set; }

    private Tensor LoadTensorFromFile(string filePath)
    {
        if (filePath == null) return null;
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"The specified file was not found: {filePath}");
        }
        return Tensor.Load(filePath)?.to(Device).to_type(Type);
    }

    /// <summary>
    /// Creates parameters for a Kalman filter model using the properties of this class.
    /// </summary>
    public IObservable<KalmanFilterParameters> Process()
    {
        Device ??= CPU;

        var transitionMatrix = LoadTensorFromFile(TransitionMatrixFilePath);
        var measurementFunction = LoadTensorFromFile(MeasurementFunctionFilePath);
        var processNoiseCovariance = LoadTensorFromFile(ProcessNoiseCovarianceFilePath);
        var measurementNoiseCovariance = LoadTensorFromFile(MeasurementNoiseCovarianceFilePath);
        var initialMean = LoadTensorFromFile(InitialMeanFilePath);
        var initialCovariance = LoadTensorFromFile(InitialCovarianceFilePath);

        var parameters = new KalmanFilterParameters(
            transitionMatrix: transitionMatrix,
            measurementFunction: measurementFunction,
            processNoiseCovariance: processNoiseCovariance,
            measurementNoiseCovariance: measurementNoiseCovariance,
            initialMean: initialMean,
            initialCovariance: initialCovariance
        );

        return Observable.Return(parameters);
    }
}