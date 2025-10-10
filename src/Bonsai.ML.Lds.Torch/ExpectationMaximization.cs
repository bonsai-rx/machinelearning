using System;
using System.ComponentModel;
using System.Reactive.Linq;
using TorchSharp;
using static TorchSharp.torch;
using System.Threading.Tasks;

namespace Bonsai.ML.Lds.Torch;

/// <summary>
/// Learn the parameters of a kalman filter using the batch EM update algorithm.
/// </summary>
[Combinator]
[Description("Learn the parameters of a kalman filter using the batch EM update algorithm.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class ExpectationMaximization
{
    /// <summary>
    /// The name of the Kalman filter model to be trained.
    /// </summary>
    [TypeConverter(typeof(KalmanFilterNameConverter))]
    [Description("The name of the Kalman filter model to be trained.")]
    public string ModelName { get; set; } = "KalmanFilter";

    private int _maxIterations = 10;
    /// <summary>
    /// The maximum number of EM iterations to perform.
    /// </summary>
    [Description("The maximum number of EM iterations to perform.")]
    public int MaxIterations
    {
        get => _maxIterations;
        set => _maxIterations = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(MaxIterations), "Must be greater than zero.");
    }

    private double _tolerance = 1e-4;
    /// <summary>
    /// The convergence tolerance for the EM algorithm.
    /// </summary>
    [Description("The convergence tolerance for the EM algorithm.")]
    public double Tolerance
    {
        get => _tolerance;
        set => _tolerance = value >= 0 ? value : throw new ArgumentOutOfRangeException(nameof(Tolerance), "Must be greater than or equal to zero.");
    }

    private bool _verbose = true;
    /// <summary>
    /// If true, prints progress messages to the console.
    /// </summary>
    [Description("If true, prints progress messages to the console.")]
    public bool Verbose
    {
        get => _verbose;
        set => _verbose = value;
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
    public IObservable<ExpectationMaximizationResult> Process(IObservable<Tensor> source)
    {
        return source.SelectMany(input => Observable.Create<ExpectationMaximizationResult>((observer, cancellationToken) =>
        {
            return Task.Run(() =>
            {
                var model = KalmanFilterModelManager.GetKalmanFilter(ModelName);
                var previousLogLikelihood = double.NegativeInfinity;
                var logLikelihood = zeros(new long[] { MaxIterations }, device: input.device);

                var parametersToEstimate = new ParametersToEstimate(
                    transitionMatrix: EstimateTransitionMatrix,
                    measurementFunction: EstimateMeasurementFunction,
                    processNoiseCovariance: EstimateProcessNoiseCovariance,
                    measurementNoiseCovariance: EstimateMeasurementNoiseCovariance,
                    initialMean: EstimateInitialMean,
                    initialCovariance: EstimateInitialCovariance);

                for (int i = 0; i < MaxIterations; i++)
                {
                    // Check for cancellation before each iteration
                    if (cancellationToken.IsCancellationRequested)
                    {
                        observer.OnCompleted();
                        return System.Reactive.Disposables.Disposable.Empty;
                    }

                    var result = model.ExpectationMaximization(input, 1, Tolerance, parametersToEstimate, false);

                    var logLikelihoodSum = result.LogLikelihood
                        .cpu()
                        .to_type(ScalarType.Float32)
                        .ReadCpuSingle(0);

                    logLikelihood[i] = logLikelihoodSum;

                    if (Verbose)
                    {
                        Console.WriteLine("Iteration " + (i + 1) + ", Log Likelihood: " + logLikelihoodSum);
                        if (i == MaxIterations - 1)
                        {
                            Console.WriteLine("EM reached the maximum number of iterations.");
                        }
                    }

                    if (logLikelihoodSum - previousLogLikelihood < Tolerance)
                    {
                        if (Verbose)
                        {
                            Console.WriteLine("EM converged after " + (i + 1) + " iterations.");
                        }
                        logLikelihood = logLikelihood[torch.TensorIndex.Slice(0, i + 1)];
                        break;
                    }
                    previousLogLikelihood = logLikelihoodSum;
                    model.UpdateParameters(result.Parameters);

                    observer.OnNext(new ExpectationMaximizationResult(
                        logLikelihood: logLikelihood[torch.TensorIndex.Slice(0, i + 1)],
                        parameters: model.Parameters,
                        finished: false));
                }

                observer.OnNext(new ExpectationMaximizationResult(
                    logLikelihood: logLikelihood,
                    parameters: model.Parameters,
                    finished: true));

                observer.OnCompleted();
                return System.Reactive.Disposables.Disposable.Empty;
            },
            cancellationToken);
        }));
    }
}