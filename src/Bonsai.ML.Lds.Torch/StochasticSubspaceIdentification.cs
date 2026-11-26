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
    /// <summary>
    /// The target number of states in the Kalman filter model.
    /// </summary>
    [Description("The target number of states in the Kalman filter model.")]
    public int? TargetNumStates { get; set; } = 2;

    /// <summary>
    /// The maximum lag to consider for the subspace identification.
    /// </summary>
    [Range(1, int.MaxValue)]
    [Editor(DesignTypes.NumericUpDownEditor, DesignTypes.UITypeEditor)]
    [Description("The maximum lag to consider for the subspace identification.")]
    public int MaxLag { get; set; } = 20;

    /// <summary>
    /// The threshold for the singular values to determine the effective number of states.
    /// </summary>
    [Range(0, 1)]
    [Description("The threshold for the singular values to determine the effective number of states.")]
    public double Threshold { get; set; } = 1e-4;

    /// <summary>
    /// If true, the transition matrix will be estimated during the SSI algorithm.
    /// </summary>
    [Description("If true, the transition matrix will be estimated during the SSI algorithm.")]
    public bool EstimateTransitionMatrix { get; set; } = true;

    /// <summary>
    /// If true, the measurement function will be estimated during the SSI algorithm.
    /// </summary>
    [Description("If true, the measurement function will be estimated during the SSI algorithm.")]
    public bool EstimateMeasurementFunction { get; set; } = true;

    /// <summary>
    /// If true, the process noise covariance will be estimated during the SSI algorithm.
    /// </summary>
    [Description("If true, the process noise covariance will be estimated during the SSI algorithm.")]
    public bool EstimateProcessNoiseCovariance { get; set; } = true;

    /// <summary>
    /// If true, the measurement noise covariance will be estimated during the SSI algorithm.
    /// </summary>
    [Description("If true, the measurement noise covariance will be estimated during the SSI algorithm.")]
    public bool EstimateMeasurementNoiseCovariance { get; set; } = true;

    /// <summary>
    /// If true, the initial mean will be estimated during the SSI algorithm.
    /// </summary>
    [Description("If true, the initial mean will be estimated during the SSI algorithm.")]
    public bool EstimateInitialMean { get; set; } = true;

    /// <summary>
    /// If true, the initial covariance will be estimated during the SSI algorithm.
    /// </summary>
    [Description("If true, the initial covariance will be estimated during the SSI algorithm.")]
    public bool EstimateInitialCovariance { get; set; } = true;

    /// <summary>
    /// If true, the state offset will be estimated during the SSI algorithm.
    /// </summary>
    [Description("If true, the state offset will be estimated during the SSI algorithm.")]
    public bool EstimateStateOffset { get; set; }

    /// <summary>
    /// If true, the observation offset will be estimated during the SSI algorithm.
    /// </summary>
    [Description("If true, the observation offset will be estimated during the SSI algorithm.")]
    public bool EstimateObservationOffset { get; set; }

    /// <summary>
    /// Processes an observable sequence of input tensors, applying the Stochastic Subspace Identification algorithm to learn the parameters of a Kalman filter model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<StochasticSubspaceIdentificationResult> Process(IObservable<Tensor> source)
    {
        return source.Select(input => Observable.Create<StochasticSubspaceIdentificationResult>(observer =>
        {
            return Task.Run(() =>
            {
                if (TargetNumStates <= 0)
                    throw new InvalidOperationException("The target number of states must be greater than zero.");

                if (MaxLag <= 0)
                    throw new InvalidOperationException("The maximum lag must be greater than zero.");

                if (Threshold < 0 || Threshold >= 1)
                    throw new InvalidOperationException("The threshold must be greater than or equal to zero and less than one.");

                var parametersToEstimate = new ParametersToEstimate(
                    transitionMatrix: EstimateTransitionMatrix,
                    measurementFunction: EstimateMeasurementFunction,
                    processNoiseCovariance: EstimateProcessNoiseCovariance,
                    measurementNoiseCovariance: EstimateMeasurementNoiseCovariance,
                    initialMean: EstimateInitialMean,
                    initialCovariance: EstimateInitialCovariance,
                    stateOffset: EstimateStateOffset,
                    observationOffset: EstimateObservationOffset);

                observer.OnNext(KalmanFilter.StochasticSubspaceIdentification(
                    observations: input,
                    targetNumStates: TargetNumStates,
                    maxLag: MaxLag,
                    threshold: Threshold,
                    parametersToEstimate: parametersToEstimate));

                observer.OnCompleted();
                return System.Reactive.Disposables.Disposable.Empty;
            });
        })).Concat();
    }
}
