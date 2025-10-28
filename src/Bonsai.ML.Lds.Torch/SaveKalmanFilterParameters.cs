using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Xml.Serialization;
using System.IO;
using Bonsai.ML.Torch;
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
    /// The path to the folder where the Kalman filter model parameters will be saved.
    /// </summary>
    [Editor("Bonsai.Design.FolderNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    [Description("The path to the folder where the Kalman filter model parameters will be saved.")]
    public string Path { get; set; } = string.Empty;

    /// <summary>
    /// If true, the contents of the folder will be overwritten if it already exists.
    /// </summary>
    [Description("If true, the contents of the folder will be overwritten if it already exists.")]
    public bool Overwrite { get; set; } = false;

    /// <summary>
    /// Specifies the type of suffix to add to the save path.
    /// The suffix is added as a subfolder to the save path.
    /// If DateTime is used, a suffix with the current date and time is added to the save path in the format, '[Path]/yyyyMMddHHmmss/...'.
    /// If Guid is used, a suffix with a unique 128-bit identifier is added to the save path in the format, '[Path]/[128-bit identifier]/...'.
    /// </summary>
    [Description("Specifies the type of suffix to add to the save path.")]
    public SuffixType AddSuffix { get; set; } = SuffixType.None;

    private void SaveKalmanFilterParametersToDisk(KalmanFilterParameters parameters)
    {
        if (string.IsNullOrEmpty(Path))
        {
            Path = Directory.GetCurrentDirectory();
        }

        var path = AddSuffix switch
        {
            SuffixType.DateTime => System.IO.Path.Combine(Path, $"{HighResolutionScheduler.Now:yyyyMMddHHmmss}"),
            SuffixType.Guid => System.IO.Path.Combine(Path, $"{Guid.NewGuid()}"),
            _ => Path
        };

        var transitionMatrixPath = System.IO.Path.Combine(path, "TransitionMatrix.bin");
        var measurementFunctionPath = System.IO.Path.Combine(path, "MeasurementFunction.bin");
        var processNoiseCovariancePath = System.IO.Path.Combine(path, "ProcessNoiseCovariance.bin");
        var measurementNoiseCovariancePath = System.IO.Path.Combine(path, "MeasurementNoiseCovariance.bin");
        var initialMeanPath = System.IO.Path.Combine(path, "InitialMean.bin");
        var initialCovariancePath = System.IO.Path.Combine(path, "InitialCovariance.bin");

        if (Directory.Exists(path))
        {
            if (!Overwrite && (
                File.Exists(transitionMatrixPath) ||
                File.Exists(measurementFunctionPath) ||
                File.Exists(processNoiseCovariancePath) ||
                File.Exists(measurementNoiseCovariancePath) ||
                File.Exists(initialMeanPath) ||
                File.Exists(initialCovariancePath))
            )
            {
                throw new InvalidOperationException("The save path already exists.");
            }
            else
            {
                if (File.Exists(transitionMatrixPath))
                    File.Delete(transitionMatrixPath);
                if (File.Exists(measurementFunctionPath))
                    File.Delete(measurementFunctionPath);
                if (File.Exists(processNoiseCovariancePath))
                    File.Delete(processNoiseCovariancePath);
                if (File.Exists(measurementNoiseCovariancePath))
                    File.Delete(measurementNoiseCovariancePath);
                if (File.Exists(initialMeanPath))
                    File.Delete(initialMeanPath);
                if (File.Exists(initialCovariancePath))
                    File.Delete(initialCovariancePath);
            }
        }

        Directory.CreateDirectory(path);

        parameters.TransitionMatrix.Save(transitionMatrixPath);
        parameters.MeasurementFunction.Save(measurementFunctionPath);
        parameters.ProcessNoiseCovariance.Save(processNoiseCovariancePath);
        parameters.MeasurementNoiseCovariance.Save(measurementNoiseCovariancePath);
        parameters.InitialMean.Save(initialMeanPath);
        parameters.InitialCovariance.Save(initialCovariancePath);
    }

    /// <summary>
    /// Processes an observable sequence of Kalman filter parameters, saving to files.
    /// </summary>
    public IObservable<KalmanFilterParameters> Process(IObservable<KalmanFilterParameters> source)
    {
        return source.Do(SaveKalmanFilterParametersToDisk);
    }

    /// <summary>
    /// Processes an observable sequence of Kalman filter models, saving their parameters to files.
    /// </summary>
    public IObservable<KalmanFilter> Process(IObservable<KalmanFilter> source)
    {
        return source.Do(model => SaveKalmanFilterParametersToDisk(model.Parameters));
    }

    /// <summary>
    /// Specifies the type of suffix to add to the save path.
    /// </summary>
    public enum SuffixType
    {
        /// <summary>
        /// No suffix is added to the save path.
        /// </summary>
        None,

        /// <summary>
        /// A suffix with the current date and time is added to the save path.
        /// </summary>
        DateTime,

        /// <summary>
        /// A suffix with a unique identifier is added to the save path.
        /// </summary>
        Guid
    }
}