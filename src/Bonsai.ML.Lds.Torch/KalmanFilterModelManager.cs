using System;
using System.Reactive.Disposables;
using System.Threading;
using System.Runtime.CompilerServices;
using System.Collections.Generic;
using static TorchSharp.torch;
using Bonsai.ML.Lds.Torch;

// <summary>
// Manages instances of the Kalman Filter model with a thread-safe locking mechanism for reading state tensors and writing parameters.
// </summary>
internal sealed class KalmanFilterModelManager
{
    private static readonly Dictionary<string, KalmanFilter> _models = new();

    public static KalmanFilter GetKalmanFilter(string name)
    {
        return _models.TryGetValue(name, out var model) ? model : throw new InvalidOperationException($"Kalman filter with name {name} not found.");
    }

    internal static KalmanFilterDisposable Reserve(
        string name,
        int numStates,
        int numObservations,
        Tensor transitionMatrix,
        Tensor measurementFunction,
        Tensor processNoiseVariance,
        Tensor measurementNoiseVariance,
        Tensor initialMean,
        Tensor initialCovariance,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        if (_models.ContainsKey(name))
        {
            throw new InvalidOperationException($"A Kalman filter with name {name} already exists.");
        }

        var kalmanFilter = new KalmanFilter(
            numStates: numStates,
            numObservations: numObservations,
            transitionMatrix: transitionMatrix,
            measurementFunction: measurementFunction,
            processNoiseVariance: processNoiseVariance,
            measurementNoiseVariance: measurementNoiseVariance,
            initialMean: initialMean,
            initialCovariance: initialCovariance,
            device: device,
            scalarType: scalarType ?? ScalarType.Float32
        );

        _models.Add(name, kalmanFilter);

        return new KalmanFilterDisposable(kalmanFilter, Disposable.Create(() =>
        {
            _models.Remove(name);
            kalmanFilter.Dispose();
        }));
    }

    internal static KalmanFilterDisposable Reserve(
        string name,
        KalmanFilterParameters parameters,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        if (_models.ContainsKey(name))
        {
            throw new InvalidOperationException($"A Kalman filter with name {name} already exists.");
        }

        var kalmanFilter = new KalmanFilter(
            parameters: parameters,
            device: device,
            scalarType: scalarType ?? ScalarType.Float32
        );

        _models.Add(name, kalmanFilter);

        return new KalmanFilterDisposable(kalmanFilter, Disposable.Create(() =>
        {
            _models.Remove(name);
            kalmanFilter.Dispose();
        }));
    }

    internal sealed class KalmanFilterDisposable(KalmanFilter model, IDisposable disposable) : IDisposable
    {
        private IDisposable? resource = disposable ?? throw new ArgumentNullException(nameof(disposable));

        public bool IsDisposed => resource is null;

        private readonly KalmanFilter model = model ?? throw new ArgumentNullException(nameof(model));

        public KalmanFilter Model => model;

        public void Dispose()
        {
            var disposable = Interlocked.Exchange(ref resource, null);
            disposable?.Dispose();
        }
    }
}

