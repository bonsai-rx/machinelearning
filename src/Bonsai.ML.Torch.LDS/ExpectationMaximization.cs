using System;
using System.ComponentModel;
using System.Reactive;
using System.Reactive.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Xml.Serialization;
using Bonsai;
using Bonsai.ML.Torch;
using Bonsai.ML.Torch.NeuralNets;
using Bonsai.Reactive;
using TorchSharp;
using TorchSharp.Modules;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Learn the parameters kalman filter using the batch EM update algorithm.
/// </summary>
[Combinator]
[ResetCombinator]
[Description("Learn the parameters kalman filter using the batch EM update algorithm.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class ExpectationMaximization
{
    [TypeConverter(typeof(KalmanFilterNameConverter))]
    public string ModelName { get; set; } = "KalmanFilter";

    private int _maxIterations = 10;
    public int MaxIterations
    {
        get => _maxIterations;
        set
        {
            if (value < 1) throw new ArgumentOutOfRangeException("MaxIterations must be at least 1.");
            _maxIterations = value;
        }
    }

    private double _tolerance = 1e-4;
    public double Tolerance
    {
        get => _tolerance;
        set
        {
            if (value < 0) throw new ArgumentOutOfRangeException("Tolerance must be non-negative.");
            _tolerance = value;
        }
    }

    private bool _verbose = true;
    public bool Verbose
    {
        get => _verbose;
        set => _verbose = value;
    }

    public IObservable<ExpectationMaximizationResult> Process(IObservable<torch.Tensor> source)
    {
        return source.Select(input =>
        {
            var model = KalmanFilterModelManager.GetKalmanFilter(ModelName);
            var previousLogLikelihood = double.NegativeInfinity;
            var logLikelihood = torch.zeros(new long[] { MaxIterations }, device: input.device);

            for (int i = 0; i < MaxIterations; i++)
            {
                ExpectationMaximizationResult result;
                using (KalmanFilterModelManager.Read(model))
                {
                    result = model.ExpectationMaximization(input, 1, Tolerance, false);
                }

                var logLikelihoodSum = result.LogLikelihood
                    .cpu()
                    .to_type(torch.ScalarType.Float32)
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