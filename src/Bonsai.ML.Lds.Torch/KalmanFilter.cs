using System;
using System.Collections.Generic;
using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

// disable missing XML comment warnings
# pragma warning disable CS1591

public class KalmanFilter : nn.Module
{
    private readonly Tensor _transitionMatrix;
    private readonly Tensor _measurementFunction;
    private readonly Tensor _initialMean;
    private readonly Tensor _initialCovariance;
    private readonly Tensor _processNoiseCovariance;
    private readonly Tensor _measurementNoiseCovariance;
    private readonly Tensor _identityStates;
    private readonly Tensor _mean;
    private readonly Tensor _covariance;
    private readonly int _numStates;
    private readonly int _numObservations;
    private readonly Device _device;
    private readonly ScalarType _scalarType;

    public KalmanFilterParameters Parameters => new(
        _transitionMatrix,
        _measurementFunction,
        _processNoiseCovariance,
        _measurementNoiseCovariance,
        _initialMean,
        _initialCovariance
    );

    public KalmanFilter(
        KalmanFilterParameters parameters,
        Device device = null,
        ScalarType scalarType = ScalarType.Float32) : base("KalmanFilter")
    {
        _device = device ?? CPU;
        _scalarType = scalarType;

        ValidateNumStates(parameters.TransitionMatrix, parameters.MeasurementFunction, parameters.InitialMean, parameters.InitialCovariance, parameters.ProcessNoiseCovariance, out _numStates);
        ValidateNumObservations(parameters.MeasurementFunction, parameters.MeasurementNoiseCovariance, out _numObservations);

        _identityStates = eye(_numStates, dtype: _scalarType, device: _device);

        _transitionMatrix = parameters.TransitionMatrix?.clone().to_type(_scalarType).to(_device).requires_grad_(false)
            ?? eye(_numStates, dtype: _scalarType, device: _device).requires_grad_(false);
        ValidateMatrix(_transitionMatrix, "Transition matrix", isSquare: true, expectedDimension1: _numStates);

        _measurementFunction = parameters.MeasurementFunction?.clone().to_type(_scalarType).to(_device).requires_grad_(false)
            ?? eye(_numObservations, _numStates, dtype: _scalarType, device: _device).requires_grad_(false);
        ValidateMatrix(_measurementFunction, "Measurement function", expectedDimension1: _numObservations, expectedDimension2: _numStates);

        _initialMean = parameters.InitialMean?.clone().to_type(_scalarType).to(_device).requires_grad_(false)
            ?? zeros(_numStates, dtype: _scalarType, device: _device).requires_grad_(false);
        ValidateVector(_initialMean, "Initial mean", expectedLength: _numStates);

        _initialCovariance = parameters.InitialCovariance?.clone().to_type(_scalarType).to(_device).requires_grad_(false)
            ?? eye(_numStates, dtype: _scalarType, device: _device).requires_grad_(false);
        ValidateMatrix(_initialCovariance, "Initial covariance", isSquare: true, expectedDimension1: _numStates);

        _processNoiseCovariance = parameters.ProcessNoiseCovariance ?? CreateCovarianceMatrix(tensor(1.0), _scalarType, _device, _numStates, "Process noise variance");
        _measurementNoiseCovariance = parameters.MeasurementNoiseCovariance ?? CreateCovarianceMatrix(tensor(1.0), _scalarType, _device, _numObservations, "Measurement noise variance");

        _mean = empty(0, dtype: _scalarType, device: _device).requires_grad_(false);
        _covariance = empty(0, dtype: _scalarType, device: _device).requires_grad_(false);
    }

    public KalmanFilter(
        int? numStates = null,
        int? numObservations = null,
        Tensor transitionMatrix = null,
        Tensor measurementFunction = null,
        Tensor initialMean = null,
        Tensor initialCovariance = null,
        Tensor processNoiseVariance = null,
        Tensor measurementNoiseVariance = null,
        Device device = null,
        ScalarType scalarType = ScalarType.Float32) : base("KalmanFilter")
    {
        _device = device ?? CPU;
        _scalarType = scalarType;

        if (numStates is null)
        {
            ValidateNumStates(transitionMatrix, measurementFunction, initialMean, initialCovariance, processNoiseVariance, out var inferredNumStates);
            _numStates = inferredNumStates;
        }
        else
            _numStates = numStates.Value > 0 ? numStates.Value : throw new ArgumentOutOfRangeException(nameof(numStates), "Number of states must be greater than zero.");

        if (numObservations is null)
        {
            ValidateNumObservations(measurementFunction, measurementNoiseVariance, out var inferredNumObservations);
            _numObservations = inferredNumObservations;
        }
        else
            _numObservations = numObservations.Value > 0 ? numObservations.Value : throw new ArgumentOutOfRangeException(nameof(numObservations), "Number of observations must be greater than zero.");

        _identityStates = eye(_numStates, dtype: _scalarType, device: _device);

        _transitionMatrix = transitionMatrix?.clone().to_type(_scalarType).to(_device).requires_grad_(false)
            ?? eye(_numStates, dtype: _scalarType, device: _device).requires_grad_(false);
        ValidateMatrix(_transitionMatrix, "Transition matrix", isSquare: true, expectedDimension1: _numStates);

        _measurementFunction = measurementFunction?.clone().to_type(_scalarType).to(_device).requires_grad_(false)
            ?? eye(_numObservations, _numStates, dtype: _scalarType, device: _device).requires_grad_(false);
        ValidateMatrix(_measurementFunction, "Measurement function", expectedDimension1: _numObservations, expectedDimension2: _numStates);

        _initialMean = initialMean?.clone().to_type(_scalarType).to(_device).requires_grad_(false)
            ?? zeros(_numStates, dtype: _scalarType, device: _device).requires_grad_(false);
        ValidateVector(_initialMean, "Initial mean", _numStates);

        _initialCovariance = initialCovariance?.clone().to_type(_scalarType).to(_device).requires_grad_(false)
            ?? eye(_numStates, dtype: _scalarType, device: _device).requires_grad_(false);
        ValidateMatrix(_initialCovariance, "Initial covariance", isSquare: true, expectedDimension1: _numStates);

        processNoiseVariance ??= tensor(1.0, dtype: _scalarType, device: _device);
        measurementNoiseVariance ??= tensor(1.0, dtype: _scalarType, device: _device);

        _processNoiseCovariance = CreateCovarianceMatrix(processNoiseVariance, _scalarType, _device, _numStates, "Process noise variance");
        _measurementNoiseCovariance = CreateCovarianceMatrix(measurementNoiseVariance, _scalarType, _device, _numObservations, "Measurement noise variance");

        _mean = empty(0, dtype: _scalarType, device: _device).requires_grad_(false);
        _covariance = empty(0, dtype: _scalarType, device: _device).requires_grad_(false);

        RegisterComponents();
    }

    private static void ValidateNumStates(Tensor transitionMatrix, Tensor measurementFunction, Tensor initialMean, Tensor initialCovariance, Tensor processNoiseCovariance, out int numStates)
    {
        if (transitionMatrix is not null)
        {
            ValidateMatrix(transitionMatrix, "Transition matrix", isSquare: true);
            numStates = (int)transitionMatrix.size(0);
        }
        else if (measurementFunction is not null)
        {
            ValidateMatrix(measurementFunction, "Measurement function");
            numStates = (int)measurementFunction.size(1);
        }
        else if (initialMean is not null)
        {
            ValidateVector(initialMean, "Initial mean");
            numStates = (int)initialMean.size(0);
        }
        else if (initialCovariance is not null)
        {
            ValidateMatrix(initialCovariance, "Initial covariance", isSquare: true);
            numStates = (int)initialCovariance.size(0);
        }
        else if (processNoiseCovariance is not null)
        {
            ValidateMatrix(processNoiseCovariance, "Process noise covariance", isSquare: true);
            numStates = (int)processNoiseCovariance.size(0);
        }
        else
        {
            throw new ArgumentException("At least one of the parameters must be provided to infer the number of states.");
        }
    }

    private static void ValidateNumObservations(Tensor measurementFunction, Tensor measurementNoiseCovariance, out int numObservations)
    {
        if (measurementFunction is not null)
        {
            ValidateMatrix(measurementFunction, "Measurement function");
            numObservations = (int)measurementFunction.size(0);
        }
        else if (measurementNoiseCovariance is not null)
        {
            ValidateMatrix(measurementNoiseCovariance, "Measurement noise covariance", isSquare: true);
            numObservations = (int)measurementNoiseCovariance.size(0);
        }
        else
        {
            throw new ArgumentException("At least one of the measurement function or measurement noise covariance must be provided to infer the number of observations.");
        }
    }

    private static void ValidateMatrix(Tensor matrix, string name, bool isSquare = false, int? expectedDimension1 = null, int? expectedDimension2 = null)
    {
        if (matrix.NumberOfElements == 0)
            throw new ArgumentException($"{name} must be a non-empty matrix.");

        if (matrix.Dimensions != 2)
            throw new ArgumentException($"{name} must be 2-dimensional.");

        if (isSquare && matrix.size(0) != matrix.size(1))
            throw new ArgumentException($"{name} must be square.");

        if (expectedDimension1.HasValue && matrix.size(0) != expectedDimension1.Value)
            throw new ArgumentException($"{name} must have {expectedDimension1.Value} rows.");

        if (expectedDimension2.HasValue && matrix.size(1) != expectedDimension2.Value)
            throw new ArgumentException($"{name} must have {expectedDimension2.Value} columns.");
    }

    private static void ValidateVector(Tensor vector, string name, int? expectedLength = null)
    {
        if (vector.NumberOfElements == 0)
            throw new ArgumentException($"{name} must be a non-empty vector.");

        if (vector.Dimensions != 1)
            throw new ArgumentException($"{name} must be a vector.");

        if (expectedLength.HasValue && vector.NumberOfElements != expectedLength.Value)
            throw new ArgumentException($"{name} must be a vector with length equal to {expectedLength.Value}.");
    }

    private static void ValidateScalar(Tensor scalar, string name)
    {
        if (scalar.NumberOfElements != 1)
            throw new ArgumentException($"{name} must be a scalar.");
    }

    private Tensor CreateCovarianceMatrix(Tensor variance, ScalarType scalarType, Device device, int dimension, string name)
    {
        ValidateScalar(variance, name);
        var scalar = variance.clone().squeeze().to_type(scalarType).to(device);
        return (scalar * eye(dimension, dtype: scalarType, device: device)).requires_grad_(false);
    }

    private readonly struct PredictedState(
        Tensor predictedMean,
        Tensor predictedCovariance)
    {
        public readonly Tensor PredictedMean = predictedMean;
        public readonly Tensor PredictedCovariance = predictedCovariance;
    }

    private PredictedState FilterPredict(
        Tensor mean,
        Tensor covariance) =>
            new(_transitionMatrix.matmul(mean),
                _transitionMatrix.matmul(covariance)
                    .matmul(_transitionMatrix.mT) + _processNoiseCovariance);

    private static PredictedState FilterPredict(
        Tensor mean,
        Tensor covariance,
        Tensor transitionMatrix,
        Tensor processNoiseCovariance) =>
            new(transitionMatrix.matmul(mean),
                transitionMatrix.matmul(covariance)
                    .matmul(transitionMatrix.mT) + processNoiseCovariance);

    private readonly struct UpdatedState(
        Tensor updatedMean,
        Tensor updatedCovariance,
        Tensor innovation,
        Tensor innovationCovariance,
        Tensor kalmanGain)
    {
        public readonly Tensor UpdatedMean = updatedMean;
        public readonly Tensor UpdatedCovariance = updatedCovariance;
        public readonly Tensor Innovation = innovation;
        public readonly Tensor InnovationCovariance = innovationCovariance;
        public readonly Tensor KalmanGain = kalmanGain;
    }

    private UpdatedState FilterUpdate(
        Tensor predictedMean,
        Tensor predictedCovariance,
        Tensor observation)
    {
        // Innovation step
        var innovation = observation - _measurementFunction.matmul(predictedMean);
        var innovationCovariance = WrappedTensorDisposeScope(() => EnsureSymmetric(
            _measurementFunction.matmul(predictedCovariance)
                .matmul(_measurementFunction.mT) + _measurementNoiseCovariance));

        // Kalman gain
        var kalmanGain = WrappedTensorDisposeScope(() => InverseCholesky(
            predictedCovariance.matmul(_measurementFunction.mT),
            innovationCovariance));

        // Update step
        var updatedMean = predictedMean + kalmanGain.matmul(innovation);
        var updatedCovariance = WrappedTensorDisposeScope(() => predictedCovariance
                - kalmanGain.matmul(_measurementFunction).matmul(predictedCovariance));

        return new UpdatedState(updatedMean, updatedCovariance, innovation, innovationCovariance, kalmanGain);
    }

    private static UpdatedState FilterUpdate(
        Tensor predictedMean,
        Tensor predictedCovariance,
        Tensor observation,
        Tensor measurementFunction,
        Tensor measurementNoiseCovariance)
    {
        // Innovation step
        var innovation = observation - measurementFunction.matmul(predictedMean);
        var innovationCovariance = WrappedTensorDisposeScope(() => EnsureSymmetric(
            measurementFunction.matmul(predictedCovariance)
                .matmul(measurementFunction.mT) + measurementNoiseCovariance));

        // Kalman gain
        var kalmanGain = WrappedTensorDisposeScope(() => InverseCholesky(
            predictedCovariance.matmul(measurementFunction.mT),
            innovationCovariance));

        // Update step
        var updatedMean = predictedMean + kalmanGain.matmul(innovation);
        var updatedCovariance = WrappedTensorDisposeScope(() => predictedCovariance
                - kalmanGain.matmul(measurementFunction).matmul(predictedCovariance));

        return new UpdatedState(
            updatedMean: updatedMean,
            updatedCovariance: updatedCovariance,
            innovation: innovation,
            innovationCovariance: innovationCovariance,
            kalmanGain: kalmanGain
        );
    }

    public FilteredState Filter(Tensor observation)
    {
        using var g = no_grad();

        var obs = observation.atleast_2d();
        var timeBins = obs.size(0);

        var predictedMean = empty(new long[] { timeBins, _numStates }, dtype: _scalarType, device: _device);
        var predictedCovariance = empty(new long[] { timeBins, _numStates, _numStates }, dtype: _scalarType, device: _device);
        var updatedMean = empty(new long[] { timeBins, _numStates }, dtype: _scalarType, device: _device);
        var updatedCovariance = empty(new long[] { timeBins, _numStates, _numStates }, dtype: _scalarType, device: _device);

        if (_mean.NumberOfElements == 0)
            _mean.set_(_initialMean);
        if (_covariance.NumberOfElements == 0)
            _covariance.set_(_initialCovariance);

        for (long time = 0; time < timeBins; time++)
        {
            // Predict
            var prediction = FilterPredict(_mean, _covariance);

            // Update
            var update = FilterUpdate(prediction.PredictedMean, prediction.PredictedCovariance, obs[time]);

            predictedMean[time] = prediction.PredictedMean;
            predictedCovariance[time] = prediction.PredictedCovariance;
            updatedMean[time] = update.UpdatedMean;
            updatedCovariance[time] = update.UpdatedCovariance;

            _mean.set_(update.UpdatedMean);
            _covariance.set_(update.UpdatedCovariance);
        }

        return new FilteredState(
            predictedMean: predictedMean,
            predictedCovariance: predictedCovariance,
            updatedMean: updatedMean,
            updatedCovariance: updatedCovariance);
    }

    private readonly struct FilteredStateWithAuxiliaryVariables(
        Tensor predictedMean,
        Tensor predictedCovariance,
        Tensor updatedMean,
        Tensor updatedCovariance,
        Tensor innovation,
        Tensor innovationCovariance,
        Tensor logLikelihood,
        Tensor kalmanGain)
    {
        public readonly Tensor PredictedMean = predictedMean;
        public readonly Tensor PredictedCovariance = predictedCovariance;
        public readonly Tensor UpdatedMean = updatedMean;
        public readonly Tensor UpdatedCovariance = updatedCovariance;
        public readonly Tensor Innovation = innovation;
        public readonly Tensor InnovationCovariance = innovationCovariance;
        public readonly Tensor LogLikelihood = logLikelihood;
        public readonly Tensor KalmanGain = kalmanGain;
    }

    private static FilteredStateWithAuxiliaryVariables Filter(
        Tensor observation,
        long timeBins,
        int numStates,
        int numObservations,
        Tensor transitionMatrix,
        Tensor measurementFunction,
        Tensor processNoiseCovariance,
        Tensor measurementNoiseCovariance,
        Tensor initialMean,
        Tensor initialCovariance,
        ScalarType scalarType,
        Device device)
    {
        var logLikelihood = empty(timeBins, dtype: scalarType, device: device);
        var predictedMean = empty(new long[] { timeBins, numStates }, dtype: scalarType, device: device);
        var predictedCovariance = empty(new long[] { timeBins, numStates, numStates }, dtype: scalarType, device: device);
        var updatedMean = empty(new long[] { timeBins, numStates }, dtype: scalarType, device: device);
        var updatedCovariance = empty(new long[] { timeBins, numStates, numStates }, dtype: scalarType, device: device);
        var innovation = empty(new long[] { timeBins, numObservations }, dtype: scalarType, device: device);
        var innovationCovariance = empty(new long[] { timeBins, numObservations, numObservations }, dtype: scalarType, device: device);
        var kalmanGain = empty(new long[] { timeBins, numStates, numObservations }, dtype: scalarType, device: device);

        var mean = initialMean;
        var covariance = initialCovariance;

        for (long time = 0; time < timeBins; time++)
        {
            // Predict
            var prediction = FilterPredict(
                mean: mean,
                covariance: covariance,
                transitionMatrix: transitionMatrix,
                processNoiseCovariance: processNoiseCovariance);

            // Update
            var update = FilterUpdate(
                predictedMean: prediction.PredictedMean,
                predictedCovariance: prediction.PredictedCovariance,
                observation: observation[time],
                measurementFunction: measurementFunction,
                measurementNoiseCovariance: measurementNoiseCovariance);

            // Log Likelihood
            var logLikelihoodData = -(slogdet(update.InnovationCovariance).logabsdet
                    + InverseCholesky(update.Innovation.T, update.InnovationCovariance)
                        .matmul(update.Innovation)).squeeze();

            // Detach and assign
            logLikelihood[time] = logLikelihoodData;
            predictedMean[time] = prediction.PredictedMean;
            predictedCovariance[time] = prediction.PredictedCovariance;
            updatedMean[time] = update.UpdatedMean;
            updatedCovariance[time] = update.UpdatedCovariance;
            innovation[time] = update.Innovation;
            innovationCovariance[time] = update.InnovationCovariance;
            kalmanGain[time] = update.KalmanGain;

            mean = update.UpdatedMean;
            covariance = update.UpdatedCovariance;
        }

        return new FilteredStateWithAuxiliaryVariables(
            predictedMean: predictedMean,
            predictedCovariance: predictedCovariance,
            updatedMean: updatedMean,
            updatedCovariance: updatedCovariance,
            innovation: innovation,
            innovationCovariance: innovationCovariance,
            logLikelihood: logLikelihood,
            kalmanGain: kalmanGain
        );
    }

    public SmoothedState Smooth(FilteredState filteredState)
    {
        using var g = no_grad();

        var predictedMean = filteredState.PredictedMean;
        var predictedCovariance = filteredState.PredictedCovariance;
        var updatedMean = filteredState.UpdatedMean;
        var updatedCovariance = filteredState.UpdatedCovariance;

        var timeBins = predictedMean.size(0);
        var smoothedMean = empty_like(updatedMean);
        var smoothedCovariance = empty_like(updatedCovariance);

        // Fix the last time point
        smoothedMean[-1] = updatedMean[-1];
        smoothedCovariance[-1] = updatedCovariance[-1];

        var smoothingGain = empty(new long[] { _numStates, _numStates }, dtype: _scalarType, device: _device);

        // Backward pass
        for (long time = timeBins - 2; time >= 0; time--)
        {
            // Smoothing gain
            smoothingGain = WrappedTensorDisposeScope(() => updatedCovariance[time].matmul(
                InverseCholesky(_transitionMatrix.mT, predictedCovariance[time + 1])
            ));

            // Smoothed mean
            smoothedMean[time] = WrappedTensorDisposeScope(() => updatedMean[time]
                + smoothingGain.matmul(
                    (smoothedMean[time + 1] - predictedMean[time + 1]).unsqueeze(-1)
                ).squeeze(-1));

            // Smoothed covariance
            smoothedCovariance[time] = WrappedTensorDisposeScope(() => updatedCovariance[time] + smoothingGain
                    .matmul(smoothedCovariance[time + 1] - predictedCovariance[time + 1])
                    .matmul(smoothingGain.mT)
            );
        }

        return new SmoothedState(
            smoothedMean,
            smoothedCovariance
        );
    }

    private readonly struct SmoothedStateWithAuxiliaryVariables(
        Tensor smoothedMean,
        Tensor smoothedCovariance,
        Tensor smoothedInitialMean,
        Tensor smoothedInitialCovariance,
        Tensor S00,
        Tensor S10,
        Tensor S11)
    {
        public readonly Tensor SmoothedMean = smoothedMean;
        public readonly Tensor SmoothedCovariance = smoothedCovariance;
        public readonly Tensor SmoothedInitialMean = smoothedInitialMean;
        public readonly Tensor SmoothedInitialCovariance = smoothedInitialCovariance;
        public readonly Tensor S00 = S00;
        public readonly Tensor S10 = S10;
        public readonly Tensor S11 = S11;
    }

    private static SmoothedStateWithAuxiliaryVariables Smooth(
        FilteredStateWithAuxiliaryVariables filteredState,
        long timeBins,
        int numStates,
        Tensor transitionMatrix,
        Tensor measurementFunction,
        Tensor initialMean,
        Tensor initialCovariance,
        Tensor identityStates,
        ScalarType scalarType,
        Device device
    )
    {
        if (timeBins < 2)
            throw new ArgumentException("Smoothing requires at least two time bins.");

        var predictedMean = filteredState.PredictedMean;
        var predictedCovariance = filteredState.PredictedCovariance;
        var updatedMean = filteredState.UpdatedMean;
        var updatedCovariance = filteredState.UpdatedCovariance;
        var kalmanGain = filteredState.KalmanGain;

        var smoothedMean = empty_like(updatedMean);
        var smoothedCovariance = empty_like(updatedCovariance);

        var S00 = zeros_like(smoothedCovariance, dtype: scalarType, device: device);
        var S10 = zeros_like(smoothedCovariance, dtype: scalarType, device: device);
        var S11 = zeros_like(smoothedCovariance, dtype: scalarType, device: device);

        // Fix the last time point
        smoothedMean[-1] = updatedMean[-1];
        smoothedCovariance[-1] = updatedCovariance[-1];
        var smoothedLagOneCovariance = WrappedTensorDisposeScope(() =>
            (identityStates - kalmanGain[-1]
                .matmul(measurementFunction))
                    .matmul(transitionMatrix)
                    .matmul(updatedCovariance[-2]));

        S11[-1] = outer(updatedMean[-1], updatedMean[-1]) + updatedCovariance[-1];

        var smoothingGain = empty([numStates, numStates], dtype: scalarType, device: device);
        var smoothingGainNext = null as Tensor;

        // Backward pass
        for (long time = timeBins - 2; time >= 0; time--)
        {
            // Smoothing gain
            smoothingGain = smoothingGainNext ?? WrappedTensorDisposeScope(() => updatedCovariance[time].matmul(
                InverseCholesky(transitionMatrix.mT, predictedCovariance[time + 1])
            ));

            // Smoothed mean
            smoothedMean[time] = WrappedTensorDisposeScope(() => updatedMean[time]
                + smoothingGain.matmul(
                    (smoothedMean[time + 1] - predictedMean[time + 1]).unsqueeze(-1)
                ).squeeze(-1));

            // Smoothed covariance
            smoothedCovariance[time] = WrappedTensorDisposeScope(() => updatedCovariance[time] + smoothingGain
                    .matmul(smoothedCovariance[time + 1] - predictedCovariance[time + 1])
                    .matmul(smoothingGain.mT)
            );

            var expectationUpdate = outer(smoothedMean[time], smoothedMean[time]) + smoothedCovariance[time];
            S11[time] = expectationUpdate;
            S00[time + 1] = expectationUpdate;
            S10[time + 1] = outer(smoothedMean[time + 1], smoothedMean[time]) + smoothedLagOneCovariance;

            // Compute next smoothing gain for lag one covariance
            if (time > 0)
            {
                smoothingGainNext = WrappedTensorDisposeScope(() => updatedCovariance[time - 1]
                    .matmul(InverseCholesky(transitionMatrix.mT, predictedCovariance[time])));

                // Smoothed lag one covariance
                smoothedLagOneCovariance = WrappedTensorDisposeScope(() => updatedCovariance[time]
                    .matmul(smoothingGainNext.mT)
                    + smoothingGain.matmul(smoothedLagOneCovariance
                        - transitionMatrix.matmul(updatedCovariance[time]))
                        .matmul(smoothingGainNext.mT));
            }
        }

        var smoothingGain0 = WrappedTensorDisposeScope(() => initialCovariance.matmul(
            InverseCholesky(transitionMatrix.mT, predictedCovariance[0])
        ));

        // Smoothed initial mean
        var smoothedInitialMean = WrappedTensorDisposeScope(() => initialMean + smoothingGain0.matmul(
            (smoothedMean[0] - predictedMean[0]).unsqueeze(-1)
        ).squeeze(-1));

        // Smoothed initial covariance
        var smoothedInitialCovariance = WrappedTensorDisposeScope(() => initialCovariance + smoothingGain0
                .matmul(smoothedCovariance[0] - predictedCovariance[0])
                .matmul(smoothingGain0.mT));

        // Smoothed lag one covariance at time 0
        smoothedLagOneCovariance = WrappedTensorDisposeScope(() => updatedCovariance[0]
            .matmul(smoothingGain0.mT)
            + smoothingGain.matmul(smoothedLagOneCovariance
                - transitionMatrix.matmul(updatedCovariance[0]))
                .matmul(smoothingGain0.mT));

        S10[0] = outer(smoothedMean[0], smoothedInitialMean) + smoothedLagOneCovariance;
        S00[0] = outer(smoothedInitialMean, smoothedInitialMean) + smoothedInitialCovariance;

        return new SmoothedStateWithAuxiliaryVariables(
            smoothedMean: smoothedMean,
            smoothedCovariance: smoothedCovariance,
            smoothedInitialMean: smoothedInitialMean,
            smoothedInitialCovariance: smoothedInitialCovariance,
            S00: S00,
            S10: S10,
            S11: S11
        );
    }

    public ExpectationMaximizationResult ExpectationMaximization(
        Tensor observation,
        int maxIterations = 100,
        double tolerance = 1e-4,
        ParametersToEstimate parametersToEstimate = new(),
        bool updateParameters = true)
    {
        var timeBins = observation.size(0);
        var logLikelihood = empty(maxIterations, dtype: ScalarType.Float32, device: _device);
        var previousLogLikelihood = double.NegativeInfinity;
        var logLikelihoodConst = -0.5 * timeBins * _numObservations * Math.Log(2.0 * Math.PI);

        var transitionMatrix = _transitionMatrix;
        var measurementFunction = _measurementFunction;
        var processNoiseCovariance = _processNoiseCovariance;
        var measurementNoiseCovariance = _measurementNoiseCovariance;
        var initialMean = _initialMean;
        var initialCovariance = _initialCovariance;

        // Precompute constant observation terms reused across EM iterations
        var observationT = observation.mT;
        var autoCorrelationObservations = observationT.matmul(observation);

        using (var _ = no_grad())
        {
            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                // Filter observations
                var filteredState = Filter(
                    observation: observation,
                    timeBins: timeBins,
                    numStates: _numStates,
                    numObservations: _numObservations,
                    transitionMatrix: transitionMatrix,
                    measurementFunction: measurementFunction,
                    processNoiseCovariance: processNoiseCovariance,
                    measurementNoiseCovariance: measurementNoiseCovariance,
                    initialMean: initialMean,
                    initialCovariance: initialCovariance,
                    scalarType: _scalarType,
                    device: _device);

                // Compute log likelihood
                var llSumDouble = filteredState.LogLikelihood.sum()
                    .to_type(ScalarType.Float64).item<double>();
                var filteredLogLikelihoodSum = logLikelihoodConst + 0.5 * llSumDouble;

                logLikelihood[iteration] = filteredLogLikelihoodSum;

                // Check for convergence
                if (filteredLogLikelihoodSum <= previousLogLikelihood)
                {
                    Console.WriteLine($"Warning: Log likelihood decreased! New: {filteredLogLikelihoodSum}, Previous: {previousLogLikelihood}");
                    break;
                }

                if (filteredLogLikelihoodSum - previousLogLikelihood < tolerance)
                    break;

                previousLogLikelihood = filteredLogLikelihoodSum;

                // Smooth the filtered results
                var smoothedState = Smooth(
                    filteredState: filteredState,
                    timeBins: timeBins,
                    numStates: _numStates,
                    transitionMatrix: transitionMatrix,
                    measurementFunction: measurementFunction,
                    initialMean: initialMean,
                    initialCovariance: initialCovariance,
                    identityStates: _identityStates,
                    scalarType: _scalarType,
                    device: _device);

                // Sufficient statistics
                var S00 = smoothedState.S00.sum([0]);
                var S11 = smoothedState.S11.sum([0]);
                var S10 = smoothedState.S10.sum([0]);

                // Compute cross-correlation between observations and smoothed states
                var crossCorrelationObservations = observationT.matmul(smoothedState.SmoothedMean);

                // Update parameters
                if (parametersToEstimate.TransitionMatrix)
                    transitionMatrix = InverseCholesky(S10, S00);

                if (parametersToEstimate.MeasurementFunction)
                    measurementFunction = InverseCholesky(crossCorrelationObservations, S11);

                if (parametersToEstimate.ProcessNoiseCovariance)
                    processNoiseCovariance = WrappedTensorDisposeScope(() =>
                        EnsureSymmetric((S11 - transitionMatrix.matmul(S10.mT)) / timeBins));

                var explainedObservationCovariance = measurementFunction.matmul(crossCorrelationObservations.mT);

                if (parametersToEstimate.MeasurementNoiseCovariance)
                    measurementNoiseCovariance = WrappedTensorDisposeScope(() =>
                        EnsureSymmetric((autoCorrelationObservations - explainedObservationCovariance - explainedObservationCovariance.mT
                            + measurementFunction.matmul(S11).matmul(measurementFunction.mT)) / timeBins));

                if (parametersToEstimate.InitialMean)
                    initialMean = smoothedState.SmoothedInitialMean;

                if (parametersToEstimate.InitialCovariance)
                    initialCovariance = smoothedState.SmoothedInitialCovariance;
            }
        }

        var updatedParameters = new KalmanFilterParameters(
            transitionMatrix: transitionMatrix,
            measurementFunction: measurementFunction,
            processNoiseCovariance: processNoiseCovariance,
            measurementNoiseCovariance: measurementNoiseCovariance,
            initialMean: initialMean,
            initialCovariance: initialCovariance
        );

        if (updateParameters)
            UpdateParameters(updatedParameters);

        return new ExpectationMaximizationResult(logLikelihood, updatedParameters);
    }

    public static ExpectationMaximizationResult ExpectationMaximization(
        Tensor observation,
        int numStates,
        int numObservations,
        KalmanFilterParameters parameters,
        int maxIterations = 100,
        double tolerance = 1e-4,
        ParametersToEstimate parametersToEstimate = new(),
        Device device = null,
        ScalarType scalarType = ScalarType.Float32)
    {
        device ??= CPU;

        var timeBins = observation.size(0);
        var logLikelihood = empty(maxIterations, dtype: ScalarType.Float32, device: device);
        var previousLogLikelihood = double.NegativeInfinity;
        var logLikelihoodConst = -0.5 * timeBins * numObservations * Math.Log(2.0 * Math.PI);

        var transitionMatrix = parameters.TransitionMatrix;
        var measurementFunction = parameters.MeasurementFunction;
        var processNoiseCovariance = parameters.ProcessNoiseCovariance;
        var measurementNoiseCovariance = parameters.MeasurementNoiseCovariance;
        var initialMean = parameters.InitialMean;
        var initialCovariance = parameters.InitialCovariance;

        var identityStates = eye(numStates, dtype: scalarType, device: device);

        // Precompute constant observation terms reused across EM iterations
        var observationT = observation.mT;
        var autoCorrelationObservations = observationT.matmul(observation);

        using (var _ = no_grad())
        {
            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                // Filter observations
                var filteredState = Filter(
                    observation: observation,
                    timeBins: timeBins,
                    numStates: numStates,
                    numObservations: numObservations,
                    transitionMatrix: transitionMatrix,
                    measurementFunction: measurementFunction,
                    processNoiseCovariance: processNoiseCovariance,
                    measurementNoiseCovariance: measurementNoiseCovariance,
                    initialMean: initialMean,
                    initialCovariance: initialCovariance,
                    scalarType: scalarType,
                    device: device);

                // Compute log likelihood (avoid creating intermediate tensors)
                var llSumDouble = filteredState.LogLikelihood.sum()
                    .to_type(ScalarType.Float64).item<double>();
                var filteredLogLikelihoodSum = logLikelihoodConst + 0.5 * llSumDouble;

                logLikelihood[iteration] = filteredLogLikelihoodSum;

                // Check for convergence
                if (filteredLogLikelihoodSum <= previousLogLikelihood)
                {
                    Console.WriteLine($"Warning: Log likelihood decreased! New: {filteredLogLikelihoodSum}, Previous: {previousLogLikelihood}");
                    break;
                }

                if (filteredLogLikelihoodSum - previousLogLikelihood < tolerance)
                    break;

                previousLogLikelihood = filteredLogLikelihoodSum;

                // Smooth the filtered results
                var smoothedState = Smooth(
                    filteredState: filteredState,
                    timeBins: timeBins,
                    numStates: numStates,
                    transitionMatrix: transitionMatrix,
                    measurementFunction: measurementFunction,
                    initialMean: initialMean,
                    initialCovariance: initialCovariance,
                    identityStates: identityStates,
                    scalarType: scalarType,
                    device: device);

                // Sufficient statistics
                var S00 = smoothedState.S00.sum([0]);
                var S11 = smoothedState.S11.sum([0]);
                var S10 = smoothedState.S10.sum([0]);

                // Replace einsum with faster matmul
                var crossCorrelationObservations = observationT.matmul(smoothedState.SmoothedMean);

                // Update parameters
                if (parametersToEstimate.TransitionMatrix)
                    transitionMatrix = InverseCholesky(S10, S00);

                if (parametersToEstimate.MeasurementFunction)
                    measurementFunction = InverseCholesky(crossCorrelationObservations, S11);

                if (parametersToEstimate.ProcessNoiseCovariance)
                    processNoiseCovariance = WrappedTensorDisposeScope(() =>
                        EnsureSymmetric((S11 - transitionMatrix.matmul(S10.mT)) / timeBins));

                var explainedObservationCovariance = measurementFunction.matmul(crossCorrelationObservations.mT);

                if (parametersToEstimate.MeasurementNoiseCovariance)
                    measurementNoiseCovariance = WrappedTensorDisposeScope(() =>
                        EnsureSymmetric((autoCorrelationObservations - explainedObservationCovariance - explainedObservationCovariance.mT
                            + measurementFunction.matmul(S11).matmul(measurementFunction.mT)) / timeBins));

                if (parametersToEstimate.InitialMean)
                    initialMean = smoothedState.SmoothedInitialMean;

                if (parametersToEstimate.InitialCovariance)
                    initialCovariance = smoothedState.SmoothedInitialCovariance;
            }
        }

        var updatedParameters = new KalmanFilterParameters(
            transitionMatrix: transitionMatrix,
            measurementFunction: measurementFunction,
            processNoiseCovariance: processNoiseCovariance,
            measurementNoiseCovariance: measurementNoiseCovariance,
            initialMean: initialMean,
            initialCovariance: initialCovariance
        );

        return new ExpectationMaximizationResult(logLikelihood, updatedParameters);
    }

    public static StochasticSubspaceIdentificationResult StochasticSubspaceIdentification(
        Tensor observations,
        int? targetNumStates = null,
        int maxLag = 20,
        double threshold = 0.01,
        ParametersToEstimate parametersToEstimate = new())
    {
        using var g = no_grad();

        var timeBins = observations.size(0);
        var numObs = observations.size(1);
        var centered = observations - observations.mean([0], keepdim: true);

        // Build Hankel matrices from observations
        var numCols = (int)(timeBins - 2 * maxLag + 1);

        if (numCols <= 0)
            throw new ArgumentException($"Number of time bins ({timeBins}) must be greater than 2*maxLag ({2 * maxLag}) for subspace identification.");

        var stride = centered.stride();
        var pastView = centered.as_strided([maxLag, numCols, numObs], [stride[0], stride[0], stride[1]]);
        var past = pastView.permute(0, 2, 1).reshape(maxLag * numObs, numCols);

        var futureView = centered.narrow(0, maxLag, timeBins - maxLag)
            .as_strided([maxLag, numCols, numObs], [stride[0], stride[0], stride[1]]);
        var future = futureView.permute(0, 2, 1).reshape(maxLag * numObs, numCols);

        // Compute the projection
        var Pp = past.matmul(past.mT);
        var projection = InverseCholesky(future.matmul(past.mT), Pp).matmul(past);

        // Compute SVD of the past observations
        var (U, S, Vt) = linalg.svd(projection, fullMatrices: false);

        // Compute the effective rank
        var effectiveRank = (S > (threshold * S[0])).to_type(ScalarType.Int64).sum().item<long>();
        var effectiveStates = Math.Max(Math.Min(effectiveRank, targetNumStates ?? effectiveRank), 1);

        var Ur = U[TensorIndex.Colon, TensorIndex.Slice(0, effectiveStates)];
        var SrSqrt = S[TensorIndex.Slice(0, effectiveStates)].diag().sqrt();
        var Vrt = Vt[TensorIndex.Slice(0, effectiveStates)];

        // Estimate observability matrix
        var observability = Ur.matmul(SrSqrt);

        // Extract measurement function from first block of observability matrix
        var measurementFunction = observability[TensorIndex.Slice(0, numObs)];

        // Estimate state sequence
        var states = SrSqrt.matmul(Vrt);

        // Estimate transition matrix using shifted states
        var statesShifted = states[TensorIndex.Colon, TensorIndex.Slice(0, numCols - 1)];
        var statesNext = states[TensorIndex.Colon, TensorIndex.Slice(1, numCols)];

        var transitionMatrix = WrappedTensorDisposeScope(() => InverseCholesky(
            statesNext.matmul(statesShifted.mT),
            statesShifted.matmul(statesShifted.mT)));

        // Estimate noise covariances using residuals
        var stateResiduals = statesNext - transitionMatrix.matmul(statesShifted);
        var processNoiseCovariance = WrappedTensorDisposeScope(() => EnsureSymmetric(stateResiduals.matmul(stateResiduals.mT) / (numCols - 1)));

        // Compute the observation residuals
        var observationPredictions = measurementFunction.matmul(states);
        var observationWindow = centered[TensorIndex.Slice(maxLag, maxLag + numCols)].mT;
        var observationResiduals = observationWindow - observationPredictions;
        var measurementNoiseCovariance = WrappedTensorDisposeScope(() => EnsureSymmetric(observationResiduals.matmul(observationResiduals.mT) / numCols));

        // Initial state estimates
        var initialMean = states[TensorIndex.Colon, 0];
        var initialCovariance = WrappedTensorDisposeScope(() => EnsureSymmetric(
            states.matmul(states.mT) / numCols));

        var parameters = new KalmanFilterParameters(
            transitionMatrix: parametersToEstimate.TransitionMatrix ? transitionMatrix : null,
            measurementFunction: parametersToEstimate.MeasurementFunction ? measurementFunction : null,
            processNoiseCovariance: parametersToEstimate.ProcessNoiseCovariance ? processNoiseCovariance : null,
            measurementNoiseCovariance: parametersToEstimate.MeasurementNoiseCovariance ? measurementNoiseCovariance : null,
            initialMean: parametersToEstimate.InitialMean ? initialMean : null,
            initialCovariance: parametersToEstimate.InitialCovariance ? initialCovariance : null
        );

        return new StochasticSubspaceIdentificationResult(
            parameters: parameters,
            effectiveStates: effectiveStates,
            singularValues: S
        );
    }

    public OrthogonalizedState OrthogonalizeMeanAndCovariance(Tensor mean, Tensor covariance)
    {
        var (_, S, Vt) = linalg.svd(_measurementFunction);
        var SVt = diag(S).matmul(Vt);

        Tensor orthogonalizedMean = null;
        if (mean is not null)
            orthogonalizedMean = matmul(mean, SVt.mT);

        Tensor orthogonalizedCovariance = null;
        if (covariance is not null)
        {
            var auxilary = matmul(SVt, covariance);
            orthogonalizedCovariance = matmul(auxilary, SVt.mT);
        }

        return new OrthogonalizedState(
            orthogonalizedMean: orthogonalizedMean,
            orthogonalizedCovariance: orthogonalizedCovariance
        );
    }

    public void UpdateParameters(KalmanFilterParameters updatedParameters)
    {
        if (updatedParameters.TransitionMatrix is not null)
            _transitionMatrix.set_(updatedParameters.TransitionMatrix);
        if (updatedParameters.MeasurementFunction is not null)
            _measurementFunction.set_(updatedParameters.MeasurementFunction);
        if (updatedParameters.ProcessNoiseCovariance is not null)
            _processNoiseCovariance.set_(updatedParameters.ProcessNoiseCovariance);
        if (updatedParameters.MeasurementNoiseCovariance is not null)
            _measurementNoiseCovariance.set_(updatedParameters.MeasurementNoiseCovariance);
        if (updatedParameters.InitialMean is not null)
            _initialMean.set_(updatedParameters.InitialMean);
        if (updatedParameters.InitialCovariance is not null)
            _initialCovariance.set_(updatedParameters.InitialCovariance);
    }

    private static Tensor EnsureSymmetric(Tensor M) => 0.5 * (M + M.mT);

    private static Tensor Ensure2D(Tensor M) => M.atleast_2d();

    private static Tensor InverseCholesky(Tensor B, Tensor A)
    {
        var L = linalg.cholesky(Ensure2D(A));
        var solT = cholesky_solve(Ensure2D(B).mT, L);
        return solT.mT;
    }
}
