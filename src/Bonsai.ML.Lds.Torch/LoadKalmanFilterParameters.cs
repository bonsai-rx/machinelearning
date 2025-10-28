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
    /// The path to the folder where the Kalman filter model parameters were saved.
    /// </summary>
    [Editor("Bonsai.Design.FolderNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    [Description("The path to the folder where the Kalman filter model parameters were saved.")]
    public string Path { get; set; } = string.Empty;

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

        if (string.IsNullOrEmpty(Path))
        {
            throw new InvalidOperationException("The save path is not specified.");
        }

        if (!Directory.Exists(Path))
        {
            throw new InvalidOperationException("The save path does not exist.");
        }

        var transitionMatrix = LoadTensorFromFile("TransitionMatrix.bin");
        var measurementFunction = LoadTensorFromFile("MeasurementFunction.bin");
        var processNoiseCovariance = LoadTensorFromFile("ProcessNoiseCovariance.bin");
        var measurementNoiseCovariance = LoadTensorFromFile("MeasurementNoiseCovariance.bin");
        var initialMean = LoadTensorFromFile("InitialMean.bin");
        var initialCovariance = LoadTensorFromFile("InitialCovariance.bin");

        return Observable.Return(KalmanFilterParameters.Initialize(
            transitionMatrix: transitionMatrix,
            measurementFunction: measurementFunction,
            processNoiseCovariance: processNoiseCovariance,
            measurementNoiseCovariance: measurementNoiseCovariance,
            initialMean: initialMean,
            initialCovariance: initialCovariance,
            device: Device,
            scalarType: Type));
    }
}