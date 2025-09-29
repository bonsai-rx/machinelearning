using System;
using System.ComponentModel;
using System.Reactive.Linq;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Learn the parameters of a kalman filter using the batch EM update algorithm.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Learn the parameters of a kalman filter using the batch EM update algorithm.")]
[WorkflowElementCategory(ElementCategory.Transform)]
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
    /// Processes an observable sequence of input tensors, applying the Expectation-Maximization algorithm to learn the parameters of a Kalman filter model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<ExpectationMaximizationResult> Process(IObservable<Tensor> source)
    {
        return source.Select(input =>
        {
            var model = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            var previousLogLikelihood = double.NegativeInfinity;
            var logLikelihood = zeros(new long[] { MaxIterations }, device: input.device);

            for (int i = 0; i < MaxIterations; i++)
            {
                ExpectationMaximizationResult result;
                using (KalmanFilterModelManager.Read(model))
                {
                    result = model.ExpectationMaximization(input, 1, Tolerance, false);
                }

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

                using (KalmanFilterModelManager.Write(model))
                {
                model.UpdateParameters(result.Parameters);
                }
            }

            var expectationMaximizationResult = new ExpectationMaximizationResult(
                logLikelihood,
                model.Parameters);

            return expectationMaximizationResult;
        });
    }
}