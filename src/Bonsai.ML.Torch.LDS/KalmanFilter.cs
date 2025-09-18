using System;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LDS;

internal class KalmanFilter : nn.Module
{
    private readonly Tensor _transitionMatrix;
    private readonly Tensor _measurementFunction;
    private readonly Tensor _initialState;
    private readonly Tensor _initialCovariance;
    private readonly Tensor _processNoiseCovariance;
    private readonly Tensor _measurementNoiseCovariance;
    private readonly Tensor _identityStates;
    private readonly Tensor _state;
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
        _initialState,
        _initialCovariance
    );

    public KalmanFilter(
        KalmanFilterParameters parameters,
        Device device = null,
        ScalarType scalarType = ScalarType.Float32) : base("KalmanFilter")
    {
        _device = device ?? CPU;
        _scalarType = scalarType;

        ValidateAndSetMatrix(parameters.TransitionMatrix, "Transition matrix", _scalarType, _device, out _transitionMatrix, out _numStates, out _, isSquare: true);
        ValidateAndSetMatrix(parameters.MeasurementFunction, "Measurement function", _scalarType, _device, out _measurementFunction, out _numObservations, out _);
        ValidateAndSetVector(parameters.InitialState, "Initial state", _scalarType, _device, out _initialState, out _, expectedLength: _numStates);
        ValidateAndSetMatrix(parameters.InitialCovariance, "Initial covariance", _scalarType, _device, out _initialCovariance, out _, out _, isSquare: true, expectedDimension1: _numStates);
        ValidateAndSetMatrix(parameters.ProcessNoiseCovariance, "Process noise covariance", _scalarType, _device, out _processNoiseCovariance, out _, out _, isSquare: true, expectedDimension1: _numStates);
        ValidateAndSetMatrix(parameters.MeasurementNoiseCovariance, "Measurement noise covariance", _scalarType, _device, out _measurementNoiseCovariance, out _, out _, isSquare: true, expectedDimension1: _numObservations);

        _identityStates = eye(_numStates, dtype: _scalarType, device: _device);

        _state = empty(_numStates, dtype: _scalarType, device: _device).requires_grad_(false);
        _covariance = empty([_numStates, _numStates], dtype: _scalarType, device: _device).requires_grad_(false);
    }

    public KalmanFilter(
        int numStates,
        int numObservations,
        Tensor transitionMatrix = null,
        Tensor measurementFunction = null,
        Tensor initialState = null,
        Tensor initialCovariance = null,
        Tensor processNoiseVariance = null,
        Tensor measurementNoiseVariance = null,
        Device device = null,
        ScalarType scalarType = ScalarType.Float32) : base("KalmanFilter")
    {
        _device = device ?? CPU;
        _scalarType = scalarType;
        _numStates = numStates;
        _numObservations = numObservations;

        _identityStates = eye(_numStates, dtype: _scalarType, device: _device);

        _transitionMatrix = transitionMatrix?.clone().to_type(_scalarType).to(_device).requires_grad_(false) 
            ?? eye(_numStates, dtype: _scalarType, device: _device);
        ValidateMatrix(_transitionMatrix, "Transition matrix", isSquare: true, expectedDimension1: _numStates);

        _measurementFunction = measurementFunction?.clone().to_type(_scalarType).to(_device).requires_grad_(false)
            ?? eye(_numObservations, _numStates, dtype: _scalarType, device: _device);
        ValidateMatrix(_measurementFunction, "Measurement function", expectedDimension1: _numObservations, expectedDimension2: _numStates);

        _initialState = initialState?.clone().to_type(_scalarType).to(_device).requires_grad_(false)
            ?? zeros(_numStates, dtype: _scalarType, device: _device);
        ValidateVector(_initialState, "Initial state", _numStates);

        _initialCovariance = initialCovariance?.clone().to_type(_scalarType).to(_device).requires_grad_(false)
            ?? eye(_numStates, dtype: _scalarType, device: _device);
        ValidateMatrix(_initialCovariance, "Initial covariance", isSquare: true, expectedDimension1: _numStates);

        _processNoiseCovariance = CreateCovarianceMatrix(processNoiseVariance, _scalarType, _device, numStates, "Process noise variance");
        _measurementNoiseCovariance = CreateCovarianceMatrix(measurementNoiseVariance, _scalarType, _device, numObservations, "Measurement noise variance");

        _state = empty(_numStates, dtype: _scalarType, device: _device).requires_grad_(false);
        _covariance = empty([_numStates, _numStates], dtype: _scalarType, device: _device).requires_grad_(false);

        RegisterComponents();
    }

    private static void ValidateAndSetMatrix(Tensor matrix, string name, ScalarType scalarType, Device device, out Tensor result, out int rows, out int columns, bool isSquare = false, int? expectedDimension1 = null, int? expectedDimension2 = null)
    {
        ValidateMatrix(matrix, name, isSquare, expectedDimension1, expectedDimension2);
        result = matrix.clone().to_type(scalarType).to(device).requires_grad_(false);
        rows = (int)matrix.size(0);
        columns = (int)matrix.size(1);
    }

    private static void ValidateAndSetVector(Tensor vector, string name, ScalarType scalarType, Device device, out Tensor result, out int length, int? expectedLength = null)
    {
        ValidateVector(vector, name, expectedLength);
        result = vector.clone().to_type(scalarType).to(device).requires_grad_(false);
        length = (int)vector.size(0);
    }

    private static void ValidateAndSetScalar(Tensor scalar, string name, ScalarType scalarType, Device device, out Tensor result)
    {
        ValidateScalar(scalar, name);
        result = scalar.clone().squeeze().to_type(scalarType).to(device).requires_grad_(false);
    }

    private static void ValidateMatrix(Tensor matrix, string name, bool isSquare = false, int? expectedDimension1 = null, int? expectedDimension2 = null)
    {
        if (matrix is null)
            throw new ArgumentException($"{name} cannot be null.");

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
        if (vector is null)
            throw new ArgumentException($"{name} cannot be null.");

        if (vector.Dimensions != 1)
            throw new ArgumentException($"{name} must be a vector.");
    
        if (expectedLength.HasValue && vector.NumberOfElements != expectedLength.Value)
            throw new ArgumentException($"{name} must be a vector with length equal to {expectedLength.Value}.");
    }

    private static void ValidateScalar(Tensor scalar, string name)
    {
        if (scalar is null)
            throw new ArgumentException($"{name} cannot be null.");

        if (scalar.NumberOfElements != 1)
            throw new ArgumentException($"{name} must be a scalar.");
    }

    private Tensor CreateCovarianceMatrix(Tensor variance, ScalarType scalarType, Device device, int dimension, string name)
    {
        ValidateAndSetScalar(variance, name, scalarType, device, out var scalar);
        return (scalar * eye(dimension, dtype: scalarType, device: device)).requires_grad_(false);
    }

    private readonly struct PredictedResult(
        Tensor predictedState,
        Tensor predictedCovariance)
    {
        public readonly Tensor PredictedState = predictedState;
        public readonly Tensor PredictedCovariance = predictedCovariance;
    }

    private PredictedResult FilterPredict(
        Tensor state,
        Tensor covariance) => 
            new(_transitionMatrix.matmul(state),
                _transitionMatrix.matmul(covariance)
                    .matmul(_transitionMatrix.mT) + _processNoiseCovariance);

    private static PredictedResult FilterPredict(
        Tensor state,
        Tensor covariance,
        Tensor transitionMatrix,
        Tensor processNoiseCovariance) => 
            new(transitionMatrix.matmul(state),
                transitionMatrix.matmul(covariance)
                    .matmul(transitionMatrix.mT) + processNoiseCovariance);

    private readonly struct UpdatedResult(
        Tensor updatedState,
        Tensor updatedCovariance,
        Tensor innovation,
        Tensor innovationCovariance,
        Tensor kalmanGain)
    {
        public readonly Tensor UpdatedState = updatedState;
        public readonly Tensor UpdatedCovariance = updatedCovariance;
        public readonly Tensor Innovation = innovation;
        public readonly Tensor InnovationCovariance = innovationCovariance;
        public readonly Tensor KalmanGain = kalmanGain;
    }

    private UpdatedResult FilterUpdate(
        Tensor predictedState,
        Tensor predictedCovariance,
        Tensor observation)
    {
        // Innovation step
        var innovation = observation - _measurementFunction.matmul(predictedState);
        var innovationCovariance = WrappedTensorDisposeScope(() => EnsureSymmetric(
            _measurementFunction.matmul(predictedCovariance)
                .matmul(_measurementFunction.mT) + _measurementNoiseCovariance));

        // Kalman gain
        var kalmanGain = WrappedTensorDisposeScope(() => InverseCholesky(
            predictedCovariance.matmul(_measurementFunction.mT),
            innovationCovariance));

        // Update step
        var updatedState = predictedState + kalmanGain.matmul(innovation);
        var updatedCovariance = WrappedTensorDisposeScope(() => predictedCovariance
                - kalmanGain.matmul(_measurementFunction).matmul(predictedCovariance));

        return new UpdatedResult(updatedState, updatedCovariance, innovation, innovationCovariance, kalmanGain);
    }
    
    private static UpdatedResult FilterUpdate(
        Tensor predictedState,
        Tensor predictedCovariance,
        Tensor observation,
        Tensor measurementFunction,
        Tensor measurementNoiseCovariance)
    {
        // Innovation step
        var innovation = observation - measurementFunction.matmul(predictedState);
        var innovationCovariance = WrappedTensorDisposeScope(() => EnsureSymmetric(
            measurementFunction.matmul(predictedCovariance)
                .matmul(measurementFunction.mT) + measurementNoiseCovariance));

        // Kalman gain
        var kalmanGain = WrappedTensorDisposeScope(() => InverseCholesky(
            predictedCovariance.matmul(measurementFunction.mT),
            innovationCovariance));

        // Update step
        var updatedState = predictedState + kalmanGain.matmul(innovation);
        var updatedCovariance = WrappedTensorDisposeScope(() => predictedCovariance
                - kalmanGain.matmul(measurementFunction).matmul(predictedCovariance));

        return new UpdatedResult(
            updatedState: updatedState,
            updatedCovariance: updatedCovariance,
            innovation: innovation,
            innovationCovariance: innovationCovariance,
            kalmanGain: kalmanGain
        );
    }

    public FilteredResult Filter(Tensor observation)
    {
        var obs = observation.atleast_2d();
        var timeBins = obs.size(0);

        var predictedState = empty(new long[] { timeBins, _numStates }, dtype: _scalarType, device: _device);
        var predictedCovariance = empty(new long[] { timeBins, _numStates, _numStates }, dtype: _scalarType, device: _device);
        var updatedState = empty(new long[] { timeBins, _numStates }, dtype: _scalarType, device: _device);
        var updatedCovariance = empty(new long[] { timeBins, _numStates, _numStates }, dtype: _scalarType, device: _device);

        if (_state.NumberOfElements == 0)
            _state.set_(_initialState);
        if (_covariance.NumberOfElements == 0)
            _covariance.set_(_initialCovariance);

        for (long time = 0; time < timeBins; time++)
        {
            // Predict
            var prediction = FilterPredict(_state, _covariance);

            // Update
            var update = FilterUpdate(prediction.PredictedState, prediction.PredictedCovariance, obs[time]);

            predictedState[time] = prediction.PredictedState;
            predictedCovariance[time] = prediction.PredictedCovariance;
            updatedState[time] = update.UpdatedState;
            updatedCovariance[time] = update.UpdatedCovariance;

            _state.set_(update.UpdatedState);
            _covariance.set_(update.UpdatedCovariance);
        }

        return new FilteredResult(
            predictedState: predictedState,
            predictedCovariance: predictedCovariance,
            updatedState: updatedState,
            updatedCovariance: updatedCovariance);
    }
    
    private readonly struct FilteredResultWithAuxiliaryVariables(
        Tensor predictedState,
        Tensor predictedCovariance,
        Tensor updatedState,
        Tensor updatedCovariance,
        Tensor innovation,
        Tensor innovationCovariance,
        Tensor logLikelihood,
        Tensor kalmanGain)
    {
        public readonly Tensor PredictedState = predictedState;
        public readonly Tensor PredictedCovariance = predictedCovariance;
        public readonly Tensor UpdatedState = updatedState;
        public readonly Tensor UpdatedCovariance = updatedCovariance;
        public readonly Tensor Innovation = innovation;
        public readonly Tensor InnovationCovariance = innovationCovariance;
        public readonly Tensor LogLikelihood = logLikelihood;
        public readonly Tensor KalmanGain = kalmanGain;
    }

    private static FilteredResultWithAuxiliaryVariables Filter(
        Tensor observation,
        long timeBins,
        int numStates,
        int numObservations,
        Tensor transitionMatrix,
        Tensor measurementFunction,
        Tensor processNoiseCovariance,
        Tensor measurementNoiseCovariance,
        Tensor initialState,
        Tensor initialCovariance,
        ScalarType scalarType,
        Device device)
    {
        var logLikelihood = empty(timeBins, dtype: scalarType, device: device);
        var predictedState = empty(new long[] { timeBins, numStates }, dtype: scalarType, device: device);
        var predictedCovariance = empty(new long[] { timeBins, numStates, numStates }, dtype: scalarType, device: device);
        var updatedState = empty(new long[] { timeBins, numStates }, dtype: scalarType, device: device);
        var updatedCovariance = empty(new long[] { timeBins, numStates, numStates }, dtype: scalarType, device: device);
        var innovation = empty(new long[] { timeBins, numObservations }, dtype: scalarType, device: device);
        var innovationCovariance = empty(new long[] { timeBins, numObservations, numObservations }, dtype: scalarType, device: device);
        var kalmanGain = empty(new long[] { timeBins, numStates, numObservations }, dtype: scalarType, device: device);

        var state = initialState;
        var covariance = initialCovariance;

        for (long time = 0; time < timeBins; time++)
        {
            // Predict
            var prediction = FilterPredict(
                state: state,
                covariance: covariance,
                transitionMatrix: transitionMatrix,
                processNoiseCovariance: processNoiseCovariance);

            // Update
            var update = FilterUpdate(
                predictedState: prediction.PredictedState,
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
            predictedState[time] = prediction.PredictedState;
            predictedCovariance[time] = prediction.PredictedCovariance;
            updatedState[time] = update.UpdatedState;
            updatedCovariance[time] = update.UpdatedCovariance;
            innovation[time] = update.Innovation;
            innovationCovariance[time] = update.InnovationCovariance;
            kalmanGain[time] = update.KalmanGain;

            state = update.UpdatedState;
            covariance = update.UpdatedCovariance;
        }

        return new FilteredResultWithAuxiliaryVariables(
            predictedState: predictedState,
            predictedCovariance: predictedCovariance,
            updatedState: updatedState,
            updatedCovariance: updatedCovariance,
            innovation: innovation,
            innovationCovariance: innovationCovariance,
            logLikelihood: logLikelihood,
            kalmanGain: kalmanGain
        );
    }

    public SmoothedResult Smooth(FilteredResult filteredResult)
    {
        var predictedState = filteredResult.PredictedState;
        var predictedCovariance = filteredResult.PredictedCovariance;
        var updatedState = filteredResult.UpdatedState;
        var updatedCovariance = filteredResult.UpdatedCovariance;

        var timeBins = predictedState.size(0);
        var smoothedState = empty_like(updatedState);
        var smoothedCovariance = empty_like(updatedCovariance);

        // Fix the last time point
        smoothedState[-1] = updatedState[-1];
        smoothedCovariance[-1] = updatedCovariance[-1];

        var smoothingGain = empty(new long[] { _numStates, _numStates }, dtype: _scalarType, device: _device);

        // Backward pass
        for (long time = timeBins - 2; time >= 0; time--)
        {
            // Smoothing gain
            smoothingGain = WrappedTensorDisposeScope(() => updatedCovariance[time].matmul(
                InverseCholesky(_transitionMatrix.mT, predictedCovariance[time + 1])
            ));

            // Smoothed state
            smoothedState[time] = WrappedTensorDisposeScope(() => updatedState[time]
                + smoothingGain.matmul(
                    (smoothedState[time + 1] - predictedState[time + 1]).unsqueeze(-1)
                ).squeeze(-1));

            // Smoothed covariance
            smoothedCovariance[time] = WrappedTensorDisposeScope(() => updatedCovariance[time] + smoothingGain
                    .matmul(smoothedCovariance[time + 1] - predictedCovariance[time + 1])
                    .matmul(smoothingGain.mT)
            );
        }

        // Smoothed initial state
        var smoothedInitialState = WrappedTensorDisposeScope(() => _initialState + smoothingGain.matmul(
            (smoothedState[0] - predictedState[0]).unsqueeze(-1)
        ).squeeze(-1));

        // Smoothed initial covariance
        var smoothedInitialCovariance = WrappedTensorDisposeScope(() => _initialCovariance[0] + smoothingGain
                .matmul(smoothedCovariance[0] - predictedCovariance[0])
                .matmul(smoothingGain.mT));

        return new SmoothedResult(
            smoothedState,
            smoothedCovariance,
            smoothedInitialState,
            smoothedInitialCovariance
        );
    }

    private readonly struct SmoothedResultWithAuxiliaryVariables(
        Tensor smoothedState,
        Tensor smoothedCovariance,
        Tensor smoothedInitialState,
        Tensor smoothedInitialCovariance,
        Tensor autoCorrelationStatesCurrent,
        Tensor crossCorrelationStates,
        Tensor autoCorrelationStatesNext)
    {
        public readonly Tensor SmoothedState = smoothedState;
        public readonly Tensor SmoothedCovariance = smoothedCovariance;
        public readonly Tensor SmoothedInitialState = smoothedInitialState;
        public readonly Tensor SmoothedInitialCovariance = smoothedInitialCovariance;
        public readonly Tensor AutoCorrelationStatesCurrent = autoCorrelationStatesCurrent;
        public readonly Tensor CrossCorrelationStates = crossCorrelationStates;
        public readonly Tensor AutoCorrelationStatesNext = autoCorrelationStatesNext;
    }

    private static SmoothedResultWithAuxiliaryVariables Smooth(
        FilteredResultWithAuxiliaryVariables filteredResult,
        long timeBins,
        int numStates,
        Tensor transitionMatrix,
        Tensor measurementFunction,
        Tensor initialState,
        Tensor initialCovariance,
        Tensor identityStates,
        ScalarType scalarType,
        Device device
    )
    {
        if (timeBins < 2)
            throw new ArgumentException("Smoothing requires at least two time bins.");
        
        var predictedState = filteredResult.PredictedState;
        var predictedCovariance = filteredResult.PredictedCovariance;
        var updatedState = filteredResult.UpdatedState;
        var updatedCovariance = filteredResult.UpdatedCovariance;
        var kalmanGain = filteredResult.KalmanGain;

        var smoothedState = empty_like(updatedState);
        var smoothedCovariance = empty_like(updatedCovariance);

        var autoCorrelationStatesCurrent = zeros_like(smoothedCovariance, dtype: scalarType, device: device);
        var crossCorrelationStates = zeros_like(smoothedCovariance, dtype: scalarType, device: device);
        var autoCorrelationStatesNext = zeros_like(smoothedCovariance, dtype: scalarType, device: device);

        // Fix the last time point
        smoothedState[-1] = updatedState[-1];
        smoothedCovariance[-1] = updatedCovariance[-1];
        var smoothedLagOneCovariance = WrappedTensorDisposeScope(() =>
            (identityStates - kalmanGain[-1]
                .matmul(measurementFunction))
                    .matmul(transitionMatrix)
                    .matmul(updatedCovariance[-2]));

        autoCorrelationStatesNext[-1] = outer(updatedState[-1], updatedState[-1]) + updatedCovariance[-1];

        var smoothingGain = empty([numStates, numStates], dtype: scalarType, device: device);
        var smoothingGainNext = null as Tensor;

        // Backward pass
        for (long time = timeBins - 2; time >= 0; time--)
        {
            // Smoothing gain
            smoothingGain = smoothingGainNext ?? WrappedTensorDisposeScope(() => updatedCovariance[time].matmul(
                InverseCholesky(transitionMatrix.mT, predictedCovariance[time + 1])
            ));

            // Smoothed state
            smoothedState[time] = WrappedTensorDisposeScope(() => updatedState[time]
                + smoothingGain.matmul(
                    (smoothedState[time + 1] - predictedState[time + 1]).unsqueeze(-1)
                ).squeeze(-1));

            // Smoothed covariance
            smoothedCovariance[time] = WrappedTensorDisposeScope(() => updatedCovariance[time] + smoothingGain
                    .matmul(smoothedCovariance[time + 1] - predictedCovariance[time + 1])
                    .matmul(smoothingGain.mT)
            );

            var expectationUpdate = outer(smoothedState[time], smoothedState[time]) + smoothedCovariance[time];
            autoCorrelationStatesNext[time] = expectationUpdate;
            autoCorrelationStatesCurrent[time + 1] = expectationUpdate;
            crossCorrelationStates[time + 1] = outer(smoothedState[time + 1], smoothedState[time]) + smoothedLagOneCovariance;

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

        // Smoothed initial state
        var smoothedInitialState = WrappedTensorDisposeScope(() => initialState + smoothingGain0.matmul(
            (smoothedState[0] - predictedState[0]).unsqueeze(-1)
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

        crossCorrelationStates[0] = outer(smoothedState[0], smoothedInitialState) + smoothedLagOneCovariance;
        autoCorrelationStatesCurrent[0] = outer(smoothedInitialState, smoothedInitialState) + smoothedInitialCovariance;

        return new SmoothedResultWithAuxiliaryVariables(
            smoothedState: smoothedState,
            smoothedCovariance: smoothedCovariance,
            smoothedInitialState: smoothedInitialState,
            smoothedInitialCovariance: smoothedInitialCovariance,
            autoCorrelationStatesCurrent: autoCorrelationStatesCurrent,
            crossCorrelationStates: crossCorrelationStates,
            autoCorrelationStatesNext: autoCorrelationStatesNext
        );
    }

    public ExpectationMaximizationResult ExpectationMaximization(
        Tensor observation,
        int maxIterations = 100,
        double tolerance = 1e-4,
        bool updateParameters = true)
    {
        var timeBins = observation.size(0);
        var logLikelihood = empty(maxIterations, dtype: ScalarType.Float32, device: _device);
        var previousLogLikelihood = double.NegativeInfinity;
        var logLikelihoodConst = -0.5 * timeBins * _numObservations * log(2.0 * Math.PI);
        var updatedParameters = Parameters;

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Filter observations
            var filteredResult = Filter(
                observation: observation,
                timeBins: timeBins,
                numStates: _numStates,
                numObservations: _numObservations,
                transitionMatrix: _transitionMatrix,
                measurementFunction: _measurementFunction,
                processNoiseCovariance: _processNoiseCovariance,
                measurementNoiseCovariance: _measurementNoiseCovariance,
                initialState: _initialState,
                initialCovariance: _initialCovariance,
                scalarType: _scalarType,
                device: _device);

            // Compute log likelihood
            var filteredLogLikelihood = logLikelihoodConst + 0.5 * filteredResult.LogLikelihood.sum();
            var filteredLogLikelihoodSum = filteredLogLikelihood.to_type(ScalarType.Float64).item<double>();

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
            var smoothedResult = Smooth(
                filteredResult: filteredResult,
                timeBins: timeBins,
                numStates: _numStates,
                transitionMatrix: _transitionMatrix,
                measurementFunction: _measurementFunction,
                initialState: _initialState,
                initialCovariance: _initialCovariance,
                identityStates: _identityStates,
                scalarType: _scalarType,
                device: _device);

            // Sufficient statistics
            var autoCorrelationStatesCurrent = smoothedResult.AutoCorrelationStatesCurrent.sum([0]);
            var autoCorrelationStatesNext = smoothedResult.AutoCorrelationStatesNext.sum([0]);
            var crossCorrelationStates = smoothedResult.CrossCorrelationStates.sum([0]);

            var crossCorrelationObservations = einsum("tp,tn->pn", observation, smoothedResult.SmoothedState);
            var autoCorrelationObservations = einsum("tp,tq->pq", observation, observation);

            // Update parameters
            var updatedTransitionMatrix = InverseCholesky(crossCorrelationStates, autoCorrelationStatesCurrent);
            var updatedMeasurementFunction = InverseCholesky(crossCorrelationObservations, autoCorrelationStatesNext);
            var updatedProcessNoiseCovariance = WrappedTensorDisposeScope(() =>
                EnsureSymmetric((autoCorrelationStatesNext - InverseCholesky(crossCorrelationStates, autoCorrelationStatesCurrent).matmul(crossCorrelationStates.T)) / timeBins));

            var CSyzT = updatedMeasurementFunction.matmul(crossCorrelationObservations.mT);
            var updatedMeasurementNoiseCovariance = WrappedTensorDisposeScope(() =>
                EnsureSymmetric((autoCorrelationObservations - CSyzT - CSyzT.mT
                    + updatedMeasurementFunction.matmul(autoCorrelationStatesNext)
                        .matmul(updatedMeasurementFunction.mT)) / timeBins));

            updatedParameters = new KalmanFilterParameters(
                transitionMatrix: updatedTransitionMatrix,
                measurementFunction: updatedMeasurementFunction,
                processNoiseCovariance: updatedProcessNoiseCovariance,
                measurementNoiseCovariance: updatedMeasurementNoiseCovariance,
                initialState: smoothedResult.SmoothedInitialState,
                initialCovariance: smoothedResult.SmoothedInitialCovariance
            );

            if (updateParameters)
                UpdateParameters(updatedParameters);
        }

        return new ExpectationMaximizationResult(logLikelihood, updatedParameters);
    }

    public OrthogonalizedResult OrthogonalizeStateAndCovariance(Tensor state, Tensor covariance)
    {
        var (U, S, Vt) = linalg.svd(_measurementFunction);
        var SVt = diag(S).matmul(Vt);

        var orthogonalizedState = einsum("tk,kj->tj", state, SVt.mT);

        var auxilary = einsum("ik,tkj->tij", SVt, covariance);
        var orthogonalizedCovariance = einsum("tij,jk->tik", auxilary, SVt.mT);

        return new OrthogonalizedResult(orthogonalizedState, orthogonalizedCovariance);
    }

    public void UpdateParameters(KalmanFilterParameters updatedParameters)
    {
        _transitionMatrix.set_(updatedParameters.TransitionMatrix);
        _measurementFunction.set_(updatedParameters.MeasurementFunction);
        _processNoiseCovariance.set_(updatedParameters.ProcessNoiseCovariance);
        _measurementNoiseCovariance.set_(updatedParameters.MeasurementNoiseCovariance);
        _initialState.set_(updatedParameters.InitialState);
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
