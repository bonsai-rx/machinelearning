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

    public KalmanFilterParameters Parameters
    {
        get
        {
            return new KalmanFilterParameters(
                _transitionMatrix,
                _measurementFunction,
                _processNoiseCovariance,
                _measurementNoiseCovariance,
                _initialState,
                _initialCovariance
            );
        }
    }

    public KalmanFilter(
        KalmanFilterParameters parameters,
        Device device = null,
        ScalarType scalarType = ScalarType.Float32) : base("KalmanFilter")
    {
        device ??= CPU;

        _device = device;
        _scalarType = scalarType;

        var transitionMatrix = parameters.TransitionMatrix;
        var measurementFunction = parameters.MeasurementFunction;
        var initialState = parameters.InitialState;
        var initialCovariance = parameters.InitialCovariance;
        var processNoiseCovariance = parameters.ProcessNoiseCovariance;
        var measurementNoiseCovariance = parameters.MeasurementNoiseCovariance;

        if (transitionMatrix is null)
        {
            throw new ArgumentException("Transition matrix cannot be null.");
        }
        else
        {
            if (transitionMatrix.Dimensions != 2 ||
                transitionMatrix.size(0) != transitionMatrix.size(1))
            {
                throw new ArgumentException("Transition matrix must be square.");
            }
            _transitionMatrix = transitionMatrix.clone().to_type(_scalarType).requires_grad_(false);
            _numStates = (int)transitionMatrix.size(0);
            _identityStates = eye(_numStates, dtype: _scalarType, device: _device);
        }

        if (measurementFunction is null)
        {
            throw new ArgumentException("Measurement function cannot be null.");
        }
        else
        {
            if (measurementFunction.Dimensions != 2 ||
                measurementFunction.size(1) != _numStates)
            {
                throw new ArgumentException("Observation matrix must have dimensions [numObservations, numStates].");
            }
            _measurementFunction = measurementFunction.clone().to_type(_scalarType).requires_grad_(false);
            _numObservations = (int)measurementFunction.size(0);
            _identityObservations = eye(_numObservations, dtype: _scalarType, device: _device);
        }

        if (initialState is null)
        {
            throw new ArgumentException("Initial state cannot be null.");
        }
        else
        {
            if (initialState.NumberOfElements != _numStates)
            {
                throw new ArgumentException("Initial state must be a vector with length equal to the number of states.");
            }
            _initialState = initialState.clone().to_type(_scalarType).requires_grad_(false);
        }

        if (initialCovariance is null)
        {
            throw new ArgumentException("Initial covariance cannot be null.");
        }
        else
        {
            if (initialCovariance.Dimensions != 2 ||
                initialCovariance.size(0) != _numStates ||
                initialCovariance.size(1) != _numStates)
            {
                throw new ArgumentException("Initial covariance must be square with dimensions equal to the number of states.");
            }
            _initialCovariance = initialCovariance.clone().to_type(_scalarType).requires_grad_(false);
        }

        if (processNoiseCovariance is null)
        {
            throw new ArgumentException("Process noise covariance cannot be null.");
        }
        else
        {
            if (processNoiseCovariance.Dimensions != 2 ||
                processNoiseCovariance.size(0) != _numStates ||
                processNoiseCovariance.size(0) != _numStates)
            {
                throw new ArgumentException("Process noise covariance must be square with dimensions equal to the number of states.");
            }
            _processNoiseCovariance = processNoiseCovariance.clone().to_type(_scalarType).requires_grad_(false);
        }

        if (measurementNoiseCovariance is null)
        {
            throw new ArgumentException("Measurement noise covariance cannot be null.");
        }
        else
        {
            if (measurementNoiseCovariance.Dimensions != 2 ||
                measurementNoiseCovariance.size(0) != _numObservations ||
                measurementNoiseCovariance.size(1) != _numObservations)
            {
                throw new ArgumentException("Measurement noise variance must be a scalar.");
            }
            _measurementNoiseCovariance = measurementNoiseCovariance.clone().to_type(_scalarType).requires_grad_(false);
        }

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
        device ??= CPU;

        _device = device;
        _scalarType = scalarType;
        _numStates = numStates;
        _identityStates = eye(numStates, dtype: _scalarType, device: _device);
        _identityObservations = eye(numObservations, dtype: _scalarType, device: _device);

        if (transitionMatrix is null)
        {
            _transitionMatrix = eye(numStates, dtype: _scalarType, device: _device);
        }
        else
        {
            if (transitionMatrix.Dimensions != 2 ||
                transitionMatrix.shape[0] != numStates ||
                transitionMatrix.shape[1] != numStates)
            {
                throw new ArgumentException("Transition matrix must be square with dimensions equal to the number of states.");
            }
            _transitionMatrix = transitionMatrix.clone().to_type(_scalarType).requires_grad_(false);
        }

        if (measurementFunction is null)
        {
            _measurementFunction = eye(numObservations, numStates, dtype: _scalarType, device: _device);
        }
        else
        {
            if (measurementFunction.Dimensions != 2 ||
                measurementFunction.shape[0] != numObservations ||
                measurementFunction.shape[1] != numStates)
            {
                throw new ArgumentException("Observation matrix must have dimensions [numObservations, numStates].");
            }
            _measurementFunction = measurementFunction.clone().to_type(_scalarType).requires_grad_(false);
        }

        if (initialState is null)
        {
            _initialState = zeros(numStates, dtype: _scalarType, device: _device);
        }
        else
        {
            if (initialState.NumberOfElements != numStates)
            {
                throw new ArgumentException("Initial state must be a vector with length equal to the number of states.");
            }
            _initialState = initialState.clone().to_type(_scalarType).requires_grad_(false);
        }

        if (initialCovariance is null)
        {
            _initialCovariance = eye(numStates, dtype: _scalarType, device: _device);
        }
        else
        {
            if (initialCovariance.Dimensions != 2 ||
                initialCovariance.shape[0] != numStates ||
                initialCovariance.shape[1] != numStates)
            {
                throw new ArgumentException("Initial covariance must be square with dimensions equal to the number of states.");
            }
            _initialCovariance = initialCovariance.clone().to_type(_scalarType).requires_grad_(false);
        }

        if (processNoiseVariance is null)
        {
            _processNoiseCovariance = eye(numStates, dtype: _scalarType, device: _device);
        }
        else
        {
            if (processNoiseVariance.NumberOfElements != 1)
            {
                throw new ArgumentException("Process noise variance must be a scalar.");
            }
            _processNoiseCovariance = (processNoiseVariance * eye(numStates, dtype: _scalarType, device: _device)).requires_grad_(false);
        }

        if (measurementNoiseVariance is null)
        {
            _measurementNoiseCovariance = eye(numObservations, dtype: _scalarType, device: _device);
        }
        else
        {
            if (measurementNoiseVariance.NumberOfElements != 1)
            {
                throw new ArgumentException("Measurement noise variance must be a scalar.");
            }
            _measurementNoiseCovariance = (measurementNoiseVariance * eye(numObservations, dtype: _scalarType, device: _device)).requires_grad_(false);
        }

        _state = _initialState.clone();
        _covariance = _initialCovariance.clone();

        RegisterComponents();
    }

    private struct PredictedResult(Tensor predictedState, Tensor predictedCovariance)
    {
        public Tensor PredictedState = predictedState;
        public Tensor PredictedCovariance = predictedCovariance;
    }

    private PredictedResult FilterPredict(
        Tensor state,
        Tensor covariance
    )
    {
        var predictedState = _transitionMatrix.matmul(state);
        var predictedCovariance = EnsureSymmetric(_transitionMatrix.matmul(covariance)
            .matmul(_transitionMatrix.mT)
                + _processNoiseCovariance);

        return new PredictedResult(predictedState, predictedCovariance);
    }

    private struct UpdatedResult(
        Tensor updatedState,
        Tensor updatedCovariance,
        Tensor innovation,
        Tensor innovationCovariance,
        Tensor kalmanGain
    )
    {
        public Tensor UpdatedState = updatedState;
        public Tensor UpdatedCovariance = updatedCovariance;
        public Tensor Innovation = innovation;
        public Tensor InnovationCovariance = innovationCovariance;
        public Tensor KalmanGain = kalmanGain;
    }

    private UpdatedResult FilterUpdate(
        Tensor predictedState,
        Tensor predictedCovariance,
        Tensor observation
    )
    {
        // Innovation step
        var innovation = observation - _measurementFunction.matmul(predictedState);
        var innovationCovariance = EnsureSymmetric(
            _measurementFunction.matmul(predictedCovariance)
                .matmul(_measurementFunction.mT)
                + _measurementNoiseCovariance);

        // Kalman gain
        var kalmanGain = InverseCholesky(
            predictedCovariance.matmul(_measurementFunction.mT),
            innovationCovariance);

        // Update step
        var updatedState = predictedState + kalmanGain.matmul(innovation);
        var updatedCovariance = EnsureSymmetric(predictedCovariance
            - kalmanGain.matmul(_measurementFunction)
                .matmul(predictedCovariance));

        return new UpdatedResult(
            updatedState,
            updatedCovariance,
            innovation,
            innovationCovariance,
            kalmanGain);
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
            using (var d = NewDisposeScope())
            {
                // Predict
                var prediction = FilterPredict(_state, _covariance);

                // Update
                var update = FilterUpdate(
                    prediction.PredictedState,
                    prediction.PredictedCovariance,
                    obs[time]
                );

                // Log Likelihood
                var invInnovationCov = InverseCholesky(_identityObservations, update.InnovationCovariance);
                var logLikelihoodData = -1.0 * (slogdet(update.InnovationCovariance).Item2
                    + update.Innovation.T.matmul(invInnovationCov)
                        .matmul(update.Innovation));

                logLikelihoodData.DetachFromDisposeScope();
                prediction.PredictedState.DetachFromDisposeScope();
                prediction.PredictedCovariance.DetachFromDisposeScope();
                update.UpdatedState.DetachFromDisposeScope();
                update.UpdatedCovariance.DetachFromDisposeScope();
                update.KalmanGain.DetachFromDisposeScope();

                logLikelihood[time] = logLikelihoodData;
                predictedState[time] = prediction.PredictedState;
                predictedCovariance[time] = prediction.PredictedCovariance;
                updatedState[time] = update.UpdatedState;
                updatedCovariance[time] = update.UpdatedCovariance;
                kalmanGain[time] = update.KalmanGain;

                _state.set_(update.UpdatedState);
                _covariance.set_(update.UpdatedCovariance);
            }
        }

        var filteredResult = new FilteredResult(
            predictedState,
            predictedCovariance,
            updatedState,
            updatedCovariance,
            logLikelihood,
            kalmanGain
        );

        return filteredResult;
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
            using (var d = NewDisposeScope())
            {
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
        bool updateParameters = true
    )
    {
        var timeBins = observation.size(0);
        var logLikelihood = empty(maxIterations, dtype: ScalarType.Float32, device: _device);
        var previousLogLikelihood = double.NegativeInfinity;
        var logLikelihoodConst = -0.5 * timeBins * _numObservations * Math.Log(2 * Math.PI);
        var updatedParameters = Parameters;

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            using (var d = NewDisposeScope())
            {
                // Filter observations
                var filterResult = Filter(observation);

                // Compute log likelihood
                var filteredLogLikelihood = logLikelihoodConst + 0.5 * filterResult.LogLikelihood.sum();
                var filteredLogLikelihoodSum = filteredLogLikelihood
                    .cpu()
                    .to_type(ScalarType.Float32)
                    .ReadCpuSingle(0);

                logLikelihood[iteration] = filteredLogLikelihoodSum;

                // Check for convergence
                if (filteredLogLikelihoodSum <= previousLogLikelihood)
                {
                    // throw new ArgumentException("Log likelihood decreased, something is wrong! New likelihood: " + filteredLogLikelihoodSum + ", Previous likelihood: " + previousLogLikelihood);
                    Console.WriteLine("Warning: Log likelihood decreased! New likelihood: " + filteredLogLikelihoodSum + ", Previous likelihood: " + previousLogLikelihood);
                    break;
                }

                if (filteredLogLikelihoodSum - previousLogLikelihood < tolerance)
                {
                    break;
                }

                previousLogLikelihood = filteredLogLikelihoodSum;

                // Smooth the filtered results
                var smoothedResult = Smooth(filterResult);

                var smoothedState = smoothedResult.SmoothedState;
                var smoothedCovariance = smoothedResult.SmoothedCovariance;
                var smoothedLagOneCovariance = smoothedResult.SmoothedLagOneCovariance;

                var smoothedInitialState = smoothedResult.SmoothedInitialState;
                var smoothedInitialCovariance = smoothedResult.SmoothedInitialCovariance;

                // Sufficient statistics
                var Ezzt = smoothedCovariance + einsum("tn,tm->tnm", smoothedState, smoothedState);
                var Ezztm1 = smoothedLagOneCovariance[torch.TensorIndex.Slice(1)]
                        + einsum("tn,tm->tnm",
                            smoothedState[torch.TensorIndex.Slice(1)],
                            smoothedState[torch.TensorIndex.Slice(0, -1)]);

                var S00 = Ezzt[torch.TensorIndex.Slice(0, -1)].sum(new long[] { 0 });
                var S10 = Ezztm1.sum(new long[] { 0 });
                var S11 = Ezzt[torch.TensorIndex.Slice(1)].sum(new long[] { 0 });

                var Syz = einsum("tp,tn->pn", observation, smoothedState);
                var Eyy = einsum("tp,tq->pq", observation, observation);

                // Update transition matrix
                var updatedTransitionMatrix = InverseCholesky(S10, S00);

                // Update measurement function
                var updatedMeasurementFunction = InverseCholesky(Syz, S11);

                // Update process noise covariance
                var updatedProcessNoiseCovariance = EnsureSymmetric(
                    (S11 - InverseCholesky(S10, S00).matmul(S10.T))
                    / timeBins
                );

                // Update measurement noise covariance
                var CSyzT = updatedMeasurementFunction.matmul(Syz.mT);
                var updatedMeasurementNoiseCovariance = EnsureSymmetric(
                    (Eyy - CSyzT - CSyzT.mT
                    + updatedMeasurementFunction.matmul(S11)
                        .matmul(updatedMeasurementFunction.mT))
                    / timeBins
                );

                // Update initial state
                var updatedInitialState = smoothedInitialState;

                // Update initial covariance
                var updatedInitialCovariance = smoothedInitialCovariance;

                updatedTransitionMatrix.DetachFromDisposeScope();
                updatedMeasurementFunction.DetachFromDisposeScope();
                updatedProcessNoiseCovariance.DetachFromDisposeScope();
                updatedMeasurementNoiseCovariance.DetachFromDisposeScope();
                updatedInitialState.DetachFromDisposeScope();
                updatedInitialCovariance.DetachFromDisposeScope();

                updatedParameters = new KalmanFilterParameters(
                    updatedTransitionMatrix,
                    updatedMeasurementFunction,
                    updatedProcessNoiseCovariance,
                    updatedMeasurementNoiseCovariance,
                    updatedInitialState,
                    updatedInitialCovariance
                );

                if (updateParameters)
                {
                    UpdateParameters(updatedParameters);
                }
            }

        }

        return new ExpectationMaximizationResult(
            logLikelihood.DetachFromDisposeScope(),
            updatedParameters
        );
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

    private static Tensor EnsureSymmetric(Tensor M)
    {
        return 0.5f * (M + M.transpose(0, 1));
    }

    private static Tensor InverseCholesky(Tensor B, Tensor A)
    {
        var L = linalg.cholesky(A);
        var solT = cholesky_solve(B.transpose(0, 1), L);
        return solT.transpose(0, 1);
    }
}
