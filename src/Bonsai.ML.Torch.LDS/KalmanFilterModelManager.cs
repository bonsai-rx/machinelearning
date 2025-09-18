using System;
using System.Reactive.Disposables;
using System.Threading;
using System.Runtime.CompilerServices;
using System.Collections.Generic;
using static TorchSharp.torch;
using Bonsai.ML.Torch.LDS;
using TorchSharp;

// <summary>
// Manages instances of the Kalman Filter in a thread-safe manner.
// </summary>
internal sealed class KalmanFilterModelManager
{
    private static readonly ConditionalWeakTable<KalmanFilter, ReaderWriterLockSlim> _moduleLocks = new();

    public static ReaderWriterLockSlim GetLock(KalmanFilter instance) =>
        _moduleLocks.GetValue(instance, _ => new ReaderWriterLockSlim(LockRecursionPolicy.NoRecursion));

    public static IDisposable Read(KalmanFilter instance)
    {
        var lockObject = GetLock(instance);
        lockObject.EnterReadLock();
        return new ManagedLock(lockObject, Mode.Read);
    }

    public static IDisposable Write(KalmanFilter instance)
    {
        var lockObject = GetLock(instance);
        lockObject.EnterWriteLock();
        return new ManagedLock(lockObject, Mode.Write);
    }

    private enum Mode
    {
        Read,
        Write
    }

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
        Tensor initialState,
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
            initialState: initialState,
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

    private readonly struct ManagedLock(
        ReaderWriterLockSlim lockObject,
        Mode mode) : IDisposable
    {
        private readonly ReaderWriterLockSlim _lockObject = lockObject;
        private readonly Mode _mode = mode;

        public void Dispose()
        {
            // Exit in the reverse mode we entered.
            switch (_mode)
            {
                case Mode.Read when _lockObject.IsReadLockHeld: _lockObject.ExitReadLock(); break;
                case Mode.Write when _lockObject.IsWriteLockHeld: _lockObject.ExitWriteLock(); break;
            }
        }
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

