using System;
using System.Linq;
using System.ComponentModel;
using System.Reactive.Linq;
using TorchSharp;
using static TorchSharp.torch;
using System.Threading.Tasks;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Learn the parameters of a kalman filter using the stochastic subspace identification method.
/// </summary>
[Combinator]
[Description("Learn the parameters of a kalman filter using the stochastic subspace identification method.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class StochasticSubspaceIdentification
{
    private int _maxLag = 20;
    /// <summary>
    /// The maximum lag to consider for the subspace identification.
    /// </summary>
    [Description("The maximum lag to consider for the subspace identification.")]
    public int MaxLag
    {
        get => _maxLag;
        set => _maxLag = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(MaxLag), "Must be greater than zero.");
    }

    private double _threshold = 1e-4;
    /// <summary>
    /// The threshold for the singular values to determine the effective number of states.
    /// </summary>
    [Description("The threshold for the singular values to determine the effective number of states.")]
    public double Threshold
    {
        get => _threshold;
        set => _threshold = value >= 0 && value < 1 ? value : throw new ArgumentOutOfRangeException(nameof(Threshold), "Must be greater than or equal to zero and less than one.");
    }

    /// <summary>
    /// If true, the transition matrix will be estimated during the EM algorithm.
    /// </summary>
    [Description("If true, the transition matrix will be estimated during the EM algorithm.")]
    public bool EstimateTransitionMatrix { get; set; } = true;

    /// <summary>
    /// If true, the measurement function will be estimated during the EM algorithm.
    /// </summary>
    [Description("If true, the measurement function will be estimated during the EM algorithm.")]
    public bool EstimateMeasurementFunction { get; set; } = true;

    /// <summary>
    /// If true, the process noise covariance will be estimated during the EM algorithm.
    /// </summary>
    [Description("If true, the process noise covariance will be estimated during the EM algorithm.")]
    public bool EstimateProcessNoiseCovariance { get; set; } = true;

    /// <summary>
    /// If true, the measurement noise covariance will be estimated during the EM algorithm.
    /// </summary>
    [Description("If true, the measurement noise covariance will be estimated during the EM algorithm.")]
    public bool EstimateMeasurementNoiseCovariance { get; set; } = true;

    /// <summary>
    /// If true, the initial mean will be estimated during the EM algorithm.
    /// </summary>
    [Description("If true, the initial mean will be estimated during the EM algorithm.")]
    public bool EstimateInitialMean { get; set; } = true;

    /// <summary>
    /// If true, the initial covariance will be estimated during the EM algorithm.
    /// </summary>
    [Description("If true, the initial covariance will be estimated during the EM algorithm.")]
    public bool EstimateInitialCovariance { get; set; } = true;

    /// <summary>
    /// Processes an observable sequence of input tensors, applying the Expectation-Maximization algorithm to learn the parameters of a Kalman filter model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<StochasticSubspaceIdentificationResult> Process(IObservable<Tensor> source)
    {
        return source.Select(input => Observable.Create<StochasticSubspaceIdentificationResult>(observer =>
        {
            return Task.Run(() =>
            {
                var parametersToEstimate = new ParametersToEstimate(
                    transitionMatrix: EstimateTransitionMatrix,
                    measurementFunction: EstimateMeasurementFunction,
                    processNoiseCovariance: EstimateProcessNoiseCovariance,
                    measurementNoiseCovariance: EstimateMeasurementNoiseCovariance,
                    initialMean: EstimateInitialMean,
                    initialCovariance: EstimateInitialCovariance);

                observer.OnNext(KalmanFilter.StochasticSubspaceIdentification(
                    observations: input,
                    maxLag: MaxLag,
                    threshold: Threshold,
                    parametersToEstimate: parametersToEstimate));

                observer.OnCompleted();
                return System.Reactive.Disposables.Disposable.Empty;
            });
        })).Concat();
    }
}