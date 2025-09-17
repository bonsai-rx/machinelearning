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
    private readonly Tensor _identityObservations;
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
        _identityObservations = eye(_numObservations, dtype: _scalarType, device: _device);

        _state = _initialState.clone();
        _covariance = _initialCovariance.clone();
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
        _identityObservations = eye(_numObservations, dtype: _scalarType, device: _device);

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

        _state = _initialState.clone();
        _covariance = _initialCovariance.clone();

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

    private readonly struct PredictedResult(Tensor predictedState, Tensor predictedCovariance)
    {
        public readonly Tensor PredictedState = predictedState;
        public readonly Tensor PredictedCovariance = predictedCovariance;
    }

    private PredictedResult FilterPredict(Tensor state, Tensor covariance) => 
        new(_transitionMatrix.matmul(state),
            EnsureSymmetric(_transitionMatrix.matmul(covariance)
                .matmul(_transitionMatrix.mT) + _processNoiseCovariance));

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

    private UpdatedResult FilterUpdate(Tensor predictedState, Tensor predictedCovariance, Tensor observation)
    {
        // Innovation step
        var innovation = observation - _measurementFunction.matmul(predictedState);
        var innovationCovariance = EnsureSymmetric(
            _measurementFunction.matmul(predictedCovariance)
                .matmul(_measurementFunction.mT) + _measurementNoiseCovariance);

        // Kalman gain
        var kalmanGain = InverseCholesky(
            predictedCovariance.matmul(_measurementFunction.mT),
            innovationCovariance);

        // Update step
        var updatedState = predictedState + kalmanGain.matmul(innovation);
        var updatedCovariance = EnsureSymmetric(predictedCovariance
            - kalmanGain.matmul(_measurementFunction).matmul(predictedCovariance));

        return new UpdatedResult(updatedState, updatedCovariance, innovation, innovationCovariance, kalmanGain);
    }

    public FilteredResult Filter(Tensor observation)
    {
        var obs = observation.atleast_2d();
        var timeBins = obs.size(0);
        
        var logLikelihood = empty(timeBins, dtype: _scalarType, device: _device);
        var predictedState = empty(new long[] { timeBins, _numStates }, dtype: _scalarType, device: _device);
        var predictedCovariance = empty(new long[] { timeBins, _numStates, _numStates }, dtype: _scalarType, device: _device);
        var updatedState = empty(new long[] { timeBins, _numStates }, dtype: _scalarType, device: _device);
        var updatedCovariance = empty(new long[] { timeBins, _numStates, _numStates }, dtype: _scalarType, device: _device);
        var kalmanGain = empty(new long[] { timeBins, _numStates, _numObservations }, dtype: _scalarType, device: _device);

        for (long time = 0; time < timeBins; time++)
        {
            using var d = NewDisposeScope();
            
            // Predict
            var prediction = FilterPredict(_state, _covariance);

            // Update
            var update = FilterUpdate(prediction.PredictedState, prediction.PredictedCovariance, obs[time]);

            // Log Likelihood
            var invInnovationCov = InverseCholesky(_identityObservations, update.InnovationCovariance);
            var logLikelihoodData = -1.0 * (slogdet(update.InnovationCovariance).logabsdet
                + update.Innovation.T.matmul(invInnovationCov).matmul(update.Innovation));

            // Detach and assign
            logLikelihood[time] = logLikelihoodData.DetachFromDisposeScope();
            predictedState[time] = prediction.PredictedState.DetachFromDisposeScope();
            predictedCovariance[time] = prediction.PredictedCovariance.DetachFromDisposeScope();
            updatedState[time] = update.UpdatedState.DetachFromDisposeScope();
            updatedCovariance[time] = update.UpdatedCovariance.DetachFromDisposeScope();
            kalmanGain[time] = update.KalmanGain.DetachFromDisposeScope();

            _state.set_(update.UpdatedState);
            _covariance.set_(update.UpdatedCovariance);
        }

        return new FilteredResult(predictedState, predictedCovariance, updatedState, updatedCovariance, logLikelihood, kalmanGain);
    }

    public SmoothedResult Smooth(FilteredResult filteredResult)
    {
        var predictedState = filteredResult.PredictedState;
        var predictedCovariance = filteredResult.PredictedCovariance;
        var updatedState = filteredResult.UpdatedState;
        var updatedCovariance = filteredResult.UpdatedCovariance;
        var kalmanGain = filteredResult.KalmanGain;

        var timeBins = predictedState.size(0);
        var smoothedState = empty_like(updatedState);
        var smoothedCovariance = empty_like(updatedCovariance);
        var smoothedLagOneCovariance = empty(new long[] { timeBins, _numStates, _numStates }, dtype: _scalarType, device: _device);

        // Fix the last time point
        smoothedState[-1] = updatedState[-1];
        smoothedCovariance[-1] = updatedCovariance[-1];
        smoothedLagOneCovariance[-1] = (_identityStates - kalmanGain[-1]
            .matmul(_measurementFunction))
                .matmul(_transitionMatrix)
                .matmul(updatedCovariance[-2]);

        var smoothingGain = empty(new long[] { _numStates, _numStates }, dtype: _scalarType, device: _device);

        // Backward pass
        for (long time = timeBins - 2; time >= 0; time--)
        {
            using var d = NewDisposeScope();
            // Smoothing gain
            smoothingGain = updatedCovariance[time].matmul(
                InverseCholesky(_transitionMatrix.mT, predictedCovariance[time + 1])
            ).DetachFromDisposeScope();

            // Smoothed state
            smoothedState[time] = updatedState[time]
                + smoothingGain.matmul(
                    (smoothedState[time + 1] - predictedState[time + 1]).unsqueeze(-1)
                ).squeeze(-1)
                .DetachFromDisposeScope();

            // Smoothed covariance
            smoothedCovariance[time] = EnsureSymmetric(
                updatedCovariance[time] + smoothingGain
                    .matmul(smoothedCovariance[time + 1] - predictedCovariance[time + 1])
                    .matmul(smoothingGain.mT)
            ).DetachFromDisposeScope();

            // Compute next smoothing gain for lag one covariance
            if (time > 0)
            {
                var smoothingGainNext = updatedCovariance[time - 1]
                    .matmul(InverseCholesky(_transitionMatrix.mT, predictedCovariance[time]));

                // Smoothed lag one covariance

                smoothedLagOneCovariance[time] = smoothedCovariance[time]
                    .matmul(smoothingGainNext.mT)
                    + smoothingGain.matmul(smoothedLagOneCovariance[time + 1]
                        - _transitionMatrix.matmul(updatedCovariance[time]))
                        .matmul(smoothingGainNext.mT)
                    .DetachFromDisposeScope();
            }
        }

        // Smoothed initial state
        var smoothedInitialState = _initialState + smoothingGain.matmul(
            (smoothedState[0] - predictedState[0]).unsqueeze(-1)
        ).squeeze(-1);

        // Smoothed initial covariance
        var smoothedInitialCovariance = EnsureSymmetric(
            _initialCovariance[0] + smoothingGain
                .matmul(smoothedCovariance[0] - predictedCovariance[0])
                .matmul(smoothingGain.mT)
        );

        // Smoothing gain at time 0
        var smoothingGain0 = _initialCovariance.matmul(
            InverseCholesky(_transitionMatrix.mT, predictedCovariance[0])
        );

        // Smoothed lag one covariance at time 0
        smoothedLagOneCovariance[0] = smoothedCovariance[0]
            .matmul(smoothingGain0.mT)
            + smoothingGain.matmul(smoothedLagOneCovariance[1]
                - _transitionMatrix.matmul(updatedCovariance[0]))
                .matmul(smoothingGain0.mT)
            .DetachFromDisposeScope();

        return new SmoothedResult(
            smoothedState,
            smoothedCovariance,
            smoothedLagOneCovariance,
            smoothedInitialState,
            smoothedInitialCovariance
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
        var logLikelihoodConst = -0.5 * timeBins * _numObservations * Math.Log(2 * Math.PI);
        var updatedParameters = Parameters;

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            using var d = NewDisposeScope();
            
            // Filter observations
            var filterResult = Filter(observation);

            // Compute log likelihood
            var filteredLogLikelihood = logLikelihoodConst + 0.5 * filterResult.LogLikelihood.sum();
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
            var smoothedResult = Smooth(filterResult);

            // Sufficient statistics
            var Ezzt = smoothedResult.SmoothedCovariance + einsum("tn,tm->tnm", smoothedResult.SmoothedState, smoothedResult.SmoothedState);
            var Ezztm1 = smoothedResult.SmoothedLagOneCovariance[torch.TensorIndex.Slice(1)]
                    + einsum("tn,tm->tnm",
                        smoothedResult.SmoothedState[torch.TensorIndex.Slice(1)],
                        smoothedResult.SmoothedState[torch.TensorIndex.Slice(0, -1)]);

            var S00 = Ezzt[torch.TensorIndex.Slice(0, -1)].sum(new long[] { 0 });
            var S10 = Ezztm1.sum(new long[] { 0 });
            var S11 = Ezzt[torch.TensorIndex.Slice(1)].sum(new long[] { 0 });

            var Syz = einsum("tp,tn->pn", observation, smoothedResult.SmoothedState);
            var Eyy = einsum("tp,tq->pq", observation, observation);

            // Update parameters
            var updatedTransitionMatrix = InverseCholesky(S10, S00).DetachFromDisposeScope();
            var updatedMeasurementFunction = InverseCholesky(Syz, S11).DetachFromDisposeScope();
            var updatedProcessNoiseCovariance = EnsureSymmetric((S11 - InverseCholesky(S10, S00).matmul(S10.T)) / timeBins).DetachFromDisposeScope();

            var CSyzT = updatedMeasurementFunction.matmul(Syz.mT);
            var updatedMeasurementNoiseCovariance = EnsureSymmetric(
                (Eyy - CSyzT - CSyzT.mT + updatedMeasurementFunction.matmul(S11).matmul(updatedMeasurementFunction.mT)) / timeBins
            ).DetachFromDisposeScope();

            updatedParameters = new KalmanFilterParameters(
                updatedTransitionMatrix,
                updatedMeasurementFunction,
                updatedProcessNoiseCovariance,
                updatedMeasurementNoiseCovariance,
                smoothedResult.SmoothedInitialState.DetachFromDisposeScope(),
                smoothedResult.SmoothedInitialCovariance.DetachFromDisposeScope()
            );

            if (updateParameters)
                UpdateParameters(updatedParameters);
        }

        return new ExpectationMaximizationResult(logLikelihood.DetachFromDisposeScope(), updatedParameters);
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

    private static Tensor EnsureSymmetric(Tensor M) => 0.5f * (M + M.transpose(0, 1));

    private static Tensor InverseCholesky(Tensor B, Tensor A)
    {
        using var d = NewDisposeScope();
        var L = linalg.cholesky(A);
        var solT = cholesky_solve(B.transpose(0, 1), L);
        return solT.transpose(0, 1).MoveToOuterDisposeScope();
    }
}
