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
/// Saves the parameters of a Kalman filter model.
/// </summary>
[Combinator]
[Description("Saves the parameters of a Kalman filter model.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class SaveKalmanFilterParameters
{
    /// <summary>
    /// Specifies the path to use for saving the transition matrix of a Kalman filter to a .bin file.
    /// </summary>
    [Description("Specifies the path to use for saving the transition matrix of a Kalman filter to a .bin file.")]
    [FileNameFilter("Binary Data (*.bin)|*.bin|All Files|*.*")]
    [Editor("Bonsai.Design.SaveFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string TransitionMatrixFilePath { get; set; } = "transition_matrix.bin";

    /// <summary>
    /// Specifies the path to use for saving the measurement function of a Kalman filter to a .bin file.
    /// </summary>
    [Description("Specifies the path to use for saving the measurement function of a Kalman filter to a .bin file.")]
    [FileNameFilter("Binary Data (*.bin)|*.bin|All Files|*.*")]
    [Editor("Bonsai.Design.SaveFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string MeasurementFunctionFilePath { get; set; } = "measurement_function.bin";

    /// <summary>
    /// Specifies the path to use for saving the process noise covariance of a Kalman filter to a .bin file.
    /// </summary>
    [Description("Specifies the path to use for saving the process noise covariance of a Kalman filter to a .bin file.")]
    [FileNameFilter("Binary Data (*.bin)|*.bin|All Files|*.*")]
    [Editor("Bonsai.Design.SaveFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string ProcessNoiseCovarianceFilePath { get; set; } = "process_noise_covariance.bin";

    /// <summary>
    /// Specifies the path to use for saving the measurement noise covariance of a Kalman filter to a .bin file.
    /// </summary>
    [Description("Specifies the path to use for saving the measurement noise covariance of a Kalman filter to a .bin file.")]
    [FileNameFilter("Binary Data (*.bin)|*.bin|All Files|*.*")]
    [Editor("Bonsai.Design.SaveFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string MeasurementNoiseCovarianceFilePath { get; set; } = "measurement_noise_covariance.bin";

    /// <summary>
    /// Specifies the path to use for saving the initial mean of a Kalman filter to a .bin file.
    /// </summary>
    [Description("Specifies the path to use for saving the initial mean of a Kalman filter to a .bin file.")]
    [FileNameFilter("Binary Data (*.bin)|*.bin|All Files|*.*")]
    [Editor("Bonsai.Design.SaveFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string InitialMeanFilePath { get; set; } = "initial_mean.bin";

    /// <summary>
    /// Specifies the path to use for saving the initial covariance of a Kalman filter to a .bin file.
    /// </summary>
    [Description("Specifies the path to use for saving the initial covariance of a Kalman filter to a .bin file.")]
    [FileNameFilter("Binary Data (*.bin)|*.bin|All Files|*.*")]
    [Editor("Bonsai.Design.SaveFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    public string InitialCovarianceFilePath { get; set; } = "initial_covariance.bin";

    private void SaveTensorToFile(Tensor tensor, string filePath)
    {
        if (filePath != null)
        {
            tensor.Save(filePath);
        }
    }

    /// <summary>
    /// Creates parameters for a Kalman filter model using the properties of this class.
    /// </summary>
    public IObservable<KalmanFilter> Process(IObservable<KalmanFilter> source)
    {
        return source.Do(model =>
        {
            var parameters = model.Parameters;
            SaveTensorToFile(parameters.TransitionMatrix, TransitionMatrixFilePath);
            SaveTensorToFile(parameters.MeasurementFunction, MeasurementFunctionFilePath);
            SaveTensorToFile(parameters.ProcessNoiseCovariance, ProcessNoiseCovarianceFilePath);
            SaveTensorToFile(parameters.MeasurementNoiseCovariance, MeasurementNoiseCovarianceFilePath);
            SaveTensorToFile(parameters.InitialMean, InitialMeanFilePath);
            SaveTensorToFile(parameters.InitialCovariance, InitialCovarianceFilePath);                
        });
    }
}