using System;
using System.ComponentModel;
using System.Reactive;
using System.Xml.Serialization;
using System.Linq;
using System.Reactive.Linq;
using TorchSharp;
using static TorchSharp.torch;
using System.Threading.Tasks;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Learns the parameters of a Kalman filter using the expectation maximization algorithm.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Learns the parameters of a Kalman filter using the expectation maximization algorithm.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class ExpectationMaximization
{
    private bool _verbose = true;

    /// <summary>
    /// Gets or sets the number of states in the Kalman filter model.
    /// </summary>
    [Description("The number of states in the Kalman filter model.")]
    public int? NumStates { get; set; }

    /// <summary>
    /// Gets or sets the Kalman filter parameters used to initialize the model.
    /// </summary>
    [Description("The Kalman filter parameters used to initialize the model.")]
    [XmlIgnore]
    public KalmanFilterParameters ModelParameters { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of EM iterations to perform.
    /// </summary>
    [Range(1, int.MaxValue)]
    [Editor(DesignTypes.NumericUpDownEditor, DesignTypes.UITypeEditor)]
    [Description("The maximum number of EM iterations to perform.")]
    public int MaxIterations { get; set; } = 10;

    /// <summary>
    /// Gets or sets the convergence tolerance for the EM algorithm.
    /// </summary>
    [Description("The convergence tolerance for the EM algorithm.")]
    public double Tolerance { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets a value indicating whether to print progress messages to the console.
    /// </summary>
    [Description("If true, prints progress messages to the console.")]
    public bool Verbose
    {
        get => _verbose;
        set => _verbose = value;
    }

    /// <summary>
    /// Gets or sets a value indicating whether to estimate the transition matrix.
    /// </summary>
    [Description("If true, the transition matrix will be estimated during the EM algorithm.")]
    public bool EstimateTransitionMatrix { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to estimate the measurement function.
    /// </summary>
    [Description("If true, the measurement function will be estimated during the EM algorithm.")]
    public bool EstimateMeasurementFunction { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to estimate the process noise covariance.
    /// </summary>
    [Description("If true, the process noise covariance will be estimated during the EM algorithm.")]
    public bool EstimateProcessNoiseCovariance { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to estimate the measurement noise covariance.
    /// </summary>
    [Description("If true, the measurement noise covariance will be estimated during the EM algorithm.")]
    public bool EstimateMeasurementNoiseCovariance { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to estimate the initial mean.
    /// </summary>
    [Description("If true, the initial mean will be estimated during the EM algorithm.")]
    public bool EstimateInitialMean { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to estimate the initial covariance.
    /// </summary>
    [Description("If true, the initial covariance will be estimated during the EM algorithm.")]
    public bool EstimateInitialCovariance { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to estimate the state offset.
    /// </summary>
    [Description("If true, the state offset will be estimated during the EM algorithm.")]
    public bool EstimateStateOffset { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether to estimate the observation offset.
    /// </summary>
    [Description("If true, the observation offset will be estimated during the EM algorithm.")]
    public bool EstimateObservationOffset { get; set; }

    /// <summary>
    /// Processes an observable sequence of input tensors, applying the Expectation-Maximization algorithm to learn the parameters of a Kalman filter model.
    /// </summary>
    /// <param name="source">The sequence of observations, each a tensor with time along the first dimension and observations along the second.</param>
    /// <returns>A sequence of results containing the log likelihood and the parameters after each iteration.</returns>
    public IObservable<ExpectationMaximizationResult> Process(IObservable<Tensor> source)
    {
        return source.Select(input => Observable.Create<ExpectationMaximizationResult>((observer, cancellationToken) =>
        {
            return Task.Run(() =>
            {
                if (MaxIterations <= 0)
                    throw new InvalidOperationException("The maximum number of iterations must be greater than zero.");

                if (Tolerance < 0)
                    throw new InvalidOperationException("The tolerance must be greater than or equal to zero.");

                var maxIterations = MaxIterations;
                var tolerance = Tolerance;

                var numObservations = (int)input.size(1);
                var previousLogLikelihood = double.NegativeInfinity;
                var logLikelihood = zeros([maxIterations], device: input.device);

                var parametersToEstimate = new ParametersToEstimate(
                    transitionMatrix: EstimateTransitionMatrix,
                    measurementFunction: EstimateMeasurementFunction,
                    processNoiseCovariance: EstimateProcessNoiseCovariance,
                    measurementNoiseCovariance: EstimateMeasurementNoiseCovariance,
                    initialMean: EstimateInitialMean,
                    initialCovariance: EstimateInitialCovariance,
                    stateOffset: EstimateStateOffset,
                    observationOffset: EstimateObservationOffset);

                var parameters = ModelParameters?.Copy() ?? new KalmanFilterParameters(
                        numStates: NumStates,
                        numObservations: numObservations,
                        scalarType: input.dtype,
                        device: input.device);

                for (int i = 0; i < maxIterations; i++)
                {
                    // Check for cancellation before each iteration
                    if (cancellationToken.IsCancellationRequested)
                    {
                        observer.OnCompleted();
                        return System.Reactive.Disposables.Disposable.Empty;
                    }

                    var result = KalmanFilter.ExpectationMaximization(
                        observation: input,
                        parameters: parameters,
                        maxIterations: 1,
                        tolerance: tolerance,
                        parametersToEstimate: parametersToEstimate);

                    var logLikelihoodSum = result.LogLikelihood
                        .to_type(ScalarType.Float32)
                        .item<float>();

                    logLikelihood[i] = logLikelihoodSum;

                    var lastIteration = i == maxIterations - 1;

                    if (Verbose)
                    {
                        Console.WriteLine("Iteration " + (i + 1) + ", Log Likelihood: " + logLikelihoodSum);
                        if (lastIteration)
                            Console.WriteLine("EM reached the maximum number of iterations.");
                    }

                    if (logLikelihoodSum - previousLogLikelihood < tolerance)
                    {
                        if (Verbose)
                            Console.WriteLine("EM converged after " + (i + 1) + " iterations.");

                        logLikelihood = logLikelihood[TensorIndex.Slice(0, i + 1)];
                        break;
                    }

                    parameters = result.Parameters;

                    if (!lastIteration)
                    {
                        previousLogLikelihood = logLikelihoodSum;

                        // The parameters keep being updated in place by the next iteration, so each intermediate result gets its own copy
                        observer.OnNext(new ExpectationMaximizationResult(
                            logLikelihood: logLikelihood[TensorIndex.Slice(0, i + 1)],
                            parameters: parameters.Copy(),
                            finished: false));
                    }
                }

                observer.OnNext(new ExpectationMaximizationResult(
                    logLikelihood: logLikelihood,
                    parameters: parameters,
                    finished: true));

                observer.OnCompleted();
                return System.Reactive.Disposables.Disposable.Empty;
            },
            cancellationToken);
        })).Concat();
    }
}
