using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using System.IO;
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
    /// The path to the folder where the Kalman filter model parameters were saved.
    /// </summary>
    [Editor("Bonsai.Design.FolderNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    [Description("The path to the folder where the Kalman filter model parameters were saved.")]
    public string Path { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the data type of the tensors.
    /// </summary>
    [Description("Gets or sets the data type of the tensors.")]
    public ScalarType? Type { get; set; } = null;

    /// <summary>
    /// Gets or sets the device to use for tensor operations.
    /// </summary>
    [XmlIgnore]
    [Description("Gets or sets the device to use for tensor operations.")]
    public Device Device { get; set; } = null;

    private static Tensor LoadTensorFromFile(string basePath, string filePath)
    {
        if (filePath == null) return null;

        filePath = System.IO.Path.Combine(basePath, filePath);

        if (File.Exists(filePath))
        {
            return Tensor.Load(filePath);
        }

        return null;
    }

    /// <summary>
    /// Creates parameters for a Kalman filter model using the properties of this class.
    /// </summary>
    public IObservable<KalmanFilterParameters> Process()
    {
        if (string.IsNullOrEmpty(Path))
        {
            throw new InvalidOperationException("The save path is not specified.");
        }

        if (!Directory.Exists(Path))
        {
            throw new InvalidOperationException("The save path does not exist.");
        }

        var transitionMatrix = LoadTensorFromFile(Path, "TransitionMatrix.bin");
        var measurementFunction = LoadTensorFromFile(Path, "MeasurementFunction.bin");
        var processNoiseCovariance = LoadTensorFromFile(Path, "ProcessNoiseCovariance.bin");
        var measurementNoiseCovariance = LoadTensorFromFile(Path, "MeasurementNoiseCovariance.bin");
        var initialMean = LoadTensorFromFile(Path, "InitialMean.bin");
        var initialCovariance = LoadTensorFromFile(Path, "InitialCovariance.bin");
        var stateOffset = LoadTensorFromFile(Path, "StateOffset.bin");
        var observationOffset = LoadTensorFromFile(Path, "ObservationOffset.bin");

        var parameters = new KalmanFilterParameters(
            transitionMatrix: transitionMatrix,
            measurementFunction: measurementFunction,
            processNoiseCovariance: processNoiseCovariance,
            measurementNoiseCovariance: measurementNoiseCovariance,
            initialMean: initialMean,
            initialCovariance: initialCovariance,
            stateOffset: stateOffset,
            observationOffset: observationOffset,
            device: Device,
            scalarType: Type);

        return Observable.Return(parameters);
    }
}
