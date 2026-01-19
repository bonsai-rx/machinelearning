using System;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Bonsai.ML.Lds.Torch;

// disable missing XML comment warnings
# pragma warning disable CS1591

public class KalmanFilter : nn.Module<Tensor, LinearDynamicalSystemState>
{
    private LinearDynamicalSystemState _state;
    public readonly KalmanFilterParameters Parameters;

    public KalmanFilter(
        KalmanFilterParameters parameters) : base("KalmanFilter")
    {
        Parameters = parameters;

        _state = new LinearDynamicalSystemState(
            Parameters.InitialMean,
            Parameters.InitialCovariance);

        RegisterComponents();
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
        Tensor stateOffset = null,
        Tensor observationOffset = null,
        Device device = null,
        ScalarType? scalarType = null,
        bool requiresGrad = false) : base("KalmanFilter")
    {
        Parameters = new KalmanFilterParameters(
            numStates,
            numObservations,
            transitionMatrix,
            measurementFunction,
            processNoiseVariance,
            measurementNoiseVariance,
            initialMean,
            initialCovariance,
            stateOffset,
            observationOffset,
            device,
            scalarType,
            requiresGrad
        );

        _state = new LinearDynamicalSystemState(
            Parameters.InitialMean,
            Parameters.InitialCovariance);

        RegisterComponents();
    }

    private static LinearDynamicalSystemState FilterPredict(
        LinearDynamicalSystemState state,
        KalmanFilterParameters parameters) =>
            new(parameters.TransitionMatrix.matmul(state.Mean) + (parameters.OffsetsProvided ? parameters.StateOffset : 0),
                parameters.TransitionMatrix.matmul(state.Covariance)
                    .matmul(parameters.TransitionMatrix.mT) + parameters.ProcessNoiseCovariance);

    private static FilteredState FilterUpdate(
        Tensor observation,
        LinearDynamicalSystemState state,
        KalmanFilterParameters parameters)
    {
        if (observation is null)
            return new FilteredState(
                predictedState: state,
                updatedState: state
            );

        // Innovation step
        var innovation = observation - (parameters.MeasurementFunction.matmul(state.Mean) + (parameters.OffsetsProvided ? parameters.ObservationOffset : 0));
        var innovationCovariance = WrappedTensorDisposeScope(() => EnsureSymmetric(
            parameters.MeasurementFunction.matmul(state.Covariance)
                .matmul(parameters.MeasurementFunction.mT) + parameters.MeasurementNoiseCovariance));

        // Kalman gain
        var kalmanGain = WrappedTensorDisposeScope(() => InverseCholesky(
            state.Covariance.matmul(parameters.MeasurementFunction.mT),
            innovationCovariance));

        // Update step
        var updatedMean = state.Mean + kalmanGain.matmul(innovation);
        var updatedCovariance = WrappedTensorDisposeScope(() => state.Covariance
            - kalmanGain.matmul(parameters.MeasurementFunction).matmul(state.Covariance));

        var updatedState = new LinearDynamicalSystemState(updatedMean, updatedCovariance);

        return new FilteredState(
            predictedState: state,
            updatedState: updatedState,
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

        // We get the scalar type and device from the parameters in the event that the observations are null (e.g. during forecasting)
        var scalarType = Parameters.ScalarType;
        var device = Parameters.Device;

        var predictedState = new LinearDynamicalSystemState(
            empty([timeBins, Parameters.NumStates], dtype: scalarType, device: device),
            empty([timeBins, Parameters.NumStates, Parameters.NumStates], dtype: scalarType, device: device));

        var updatedState = new LinearDynamicalSystemState(
            empty([timeBins, Parameters.NumStates], dtype: scalarType, device: device),
            empty([timeBins, Parameters.NumStates, Parameters.NumStates], dtype: scalarType, device: device));

        for (long time = 0; time < timeBins; time++)
        {
            // Predict
            var prediction = FilterPredict(_state, Parameters);

            // Update
            var update = FilterUpdate(obs[time], prediction, Parameters);

            predictedState.Mean[time] = prediction.Mean;
            predictedState.Covariance[time] = prediction.Covariance;
            updatedState.Mean[time] = update.UpdatedState.Mean;
            updatedState.Covariance[time] = update.UpdatedState.Covariance;

            _state = update.UpdatedState;
        }

        return new FilteredState(
            predictedState: predictedState,
            updatedState: updatedState
        );
    }

    public override LinearDynamicalSystemState forward(Tensor input)
    {
        var filteredState = Filter(input);
        return filteredState.UpdatedState;
    }

    private static FilteredState Filter(
        long timeBins,
        Tensor observation,
        KalmanFilterParameters parameters)
    {
        using var g = no_grad();

        var timeBinsObservations = observation.size(0);
        timeBins = Math.Min(timeBins, timeBinsObservations);

        var filteredState = new FilteredState(
            predictedState: new LinearDynamicalSystemState(
                empty([timeBins, parameters.NumStates], dtype: parameters.ScalarType, device: parameters.Device),
                empty([timeBins, parameters.NumStates, parameters.NumStates], dtype: parameters.ScalarType, device: parameters.Device)),
            updatedState: new LinearDynamicalSystemState(
                empty([timeBins, parameters.NumStates], dtype: parameters.ScalarType, device: parameters.Device),
                empty([timeBins, parameters.NumStates, parameters.NumStates], dtype: parameters.ScalarType, device: parameters.Device)),
            innovation: empty([timeBins, parameters.NumObservations], dtype: parameters.ScalarType, device: parameters.Device),
            innovationCovariance: empty([timeBins, parameters.NumObservations, parameters.NumObservations], dtype: parameters.ScalarType, device: parameters.Device),
            kalmanGain: empty([timeBins, parameters.NumStates, parameters.NumObservations], dtype: parameters.ScalarType, device: parameters.Device),
            logLikelihood: empty([timeBins], dtype: parameters.ScalarType, device: parameters.Device)
        );

        var state = new LinearDynamicalSystemState(
            mean: parameters.InitialMean,
            covariance: parameters.InitialCovariance);

        for (long time = 0; time < timeBins; time++)
        {
            // Predict
            var prediction = FilterPredict(
                state: state,
                parameters: parameters);

            // Update
            var update = FilterUpdate(
                observation: observation[time],
                state: prediction,
                parameters: parameters);

            // Log Likelihood
            var logLikelihoodData = -(slogdet(update.InnovationCovariance).logabsdet
                    + InverseCholesky(update.Innovation.T, update.InnovationCovariance)
                        .matmul(update.Innovation)).squeeze();

            // Detach and assign
            filteredState.LogLikelihood[time] = logLikelihoodData;
            filteredState.PredictedState.Mean[time] = prediction.Mean;
            filteredState.PredictedState.Covariance[time] = prediction.Covariance;
            filteredState.UpdatedState.Mean[time] = update.UpdatedState.Mean;
            filteredState.UpdatedState.Covariance[time] = update.UpdatedState.Covariance;
            filteredState.Innovation[time] = update.Innovation;
            filteredState.InnovationCovariance[time] = update.InnovationCovariance;
            filteredState.KalmanGain[time] = update.KalmanGain;

            state = update.UpdatedState;
        }

        return filteredState;
    }

    public LinearDynamicalSystemState Smooth(FilteredState filteredState)
    {
        using var g = no_grad();

        var predictedMean = filteredState.PredictedState.Mean;
        var predictedCovariance = filteredState.PredictedState.Covariance;
        var updatedMean = filteredState.UpdatedState.Mean;
        var updatedCovariance = filteredState.UpdatedState.Covariance;

        var timeBins = predictedMean.size(0);
        var smoothedMean = empty_like(updatedMean);
        var smoothedCovariance = empty_like(updatedCovariance);

        // Fix the last time point
        smoothedMean[-1] = updatedMean[-1];
        smoothedCovariance[-1] = updatedCovariance[-1];

        var smoothingGain = empty([Parameters.NumStates, Parameters.NumStates], dtype: Parameters.ScalarType, device: Parameters.Device);

        // Backward pass
        for (long time = timeBins - 2; time >= 0; time--)
        {
            // Smoothing gain
            smoothingGain = WrappedTensorDisposeScope(() => updatedCovariance[time].matmul(
                InverseCholesky(Parameters.TransitionMatrix.mT, predictedCovariance[time + 1])
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

        return new LinearDynamicalSystemState(
            smoothedMean,
            smoothedCovariance
        );
    }

    private readonly struct SummaryStatisticsEM(
        Tensor sxx11,
        Tensor sxx10,
        Tensor sxx00,
        Tensor tx1 = null,
        Tensor tx0 = null,
        Tensor ty1 = null,
        Tensor tyx11 = null,
        Tensor tyy11 = null
    )
    {
        public readonly Tensor Sxx11 => sxx11;
        public readonly Tensor Sxx10 => sxx10;
        public readonly Tensor Sxx00 => sxx00;
        public readonly Tensor Tx1 => tx1;
        public readonly Tensor Tx0 => tx0;
        public readonly Tensor Ty1 => ty1;
        public readonly Tensor Tyx11 => tyx11;
        public readonly Tensor Tyy11 => tyy11;
    }

    private readonly struct SmoothedStateWithAuxiliaryVariables(
        LinearDynamicalSystemState smoothedState,
        Tensor smoothedInitialMean,
        Tensor smoothedInitialCovariance)
    {
        public readonly LinearDynamicalSystemState SmoothedState => smoothedState;
        public readonly Tensor SmoothedInitialMean => smoothedInitialMean;
        public readonly Tensor SmoothedInitialCovariance => smoothedInitialCovariance;
    }

    private static (SmoothedStateWithAuxiliaryVariables state, SummaryStatisticsEM statistics) Smooth(
        FilteredState filteredState,
        Tensor observations,
        KalmanFilterParameters parameters
    )
    {
        var timeBins = filteredState.PredictedState.Mean.size(0);

        if (timeBins < 2)
            throw new ArgumentException("Smoothing requires at least two time bins.");

        var predictedMean = filteredState.PredictedState.Mean;
        var predictedCovariance = filteredState.PredictedState.Covariance;
        var updatedMean = filteredState.UpdatedState.Mean;
        var updatedCovariance = filteredState.UpdatedState.Covariance;
        var kalmanGain = filteredState.KalmanGain;

        var smoothedState = new LinearDynamicalSystemState(
            mean: empty_like(updatedMean),
            covariance: empty_like(updatedCovariance)
        );

        var sxx00 = zeros_like(smoothedState.Covariance, dtype: parameters.ScalarType, device: parameters.Device);
        var sxx10 = zeros_like(smoothedState.Covariance, dtype: parameters.ScalarType, device: parameters.Device);
        var sxx11 = zeros_like(smoothedState.Covariance, dtype: parameters.ScalarType, device: parameters.Device);

        var tx1 = zeros([parameters.NumStates], dtype: parameters.ScalarType, device: parameters.Device);
        var tx0 = zeros([parameters.NumStates], dtype: parameters.ScalarType, device: parameters.Device);
        var ty1 = zeros([parameters.NumObservations], dtype: parameters.ScalarType, device: parameters.Device);
        var tyx11 = zeros([parameters.NumObservations, parameters.NumStates], dtype: parameters.ScalarType, device: parameters.Device);
        var tyy11 = zeros([parameters.NumObservations, parameters.NumObservations], dtype: parameters.ScalarType, device: parameters.Device);

        var identityStates = eye(parameters.NumStates, dtype: parameters.ScalarType, device: parameters.Device);

        // Fix the last time point
        smoothedState.Mean[-1] = updatedMean[-1];
        smoothedState.Covariance[-1] = updatedCovariance[-1];
        var smoothedLagOneCovariance = WrappedTensorDisposeScope(() =>
            (identityStates - kalmanGain[-1]
                .matmul(parameters.MeasurementFunction))
                    .matmul(parameters.TransitionMatrix)
                    .matmul(updatedCovariance[-2]));

        sxx11[-1] = outer(smoothedState.Mean[-1], smoothedState.Mean[-1]) + smoothedState.Covariance[-1];

        if (parameters.OffsetsProvided)
        {
            tx1 += smoothedState.Mean[-1];
            ty1 += observations[-1];
            tyx11 += outer(observations[-1], smoothedState.Mean[-1]);
            tyy11 += outer(observations[-1], observations[-1]);
        }

        var smoothingGain = empty([parameters.NumStates, parameters.NumStates], dtype: parameters.ScalarType, device: parameters.Device);
        var smoothingGainNext = null as Tensor;

        // Backward pass
        for (long time = timeBins - 2; time >= 0; time--)
        {
            // Smoothing gain
            smoothingGain = smoothingGainNext ?? WrappedTensorDisposeScope(() => updatedCovariance[time].matmul(
                InverseCholesky(parameters.TransitionMatrix.mT, predictedCovariance[time + 1])
            ));

            // Smoothed mean
            smoothedState.Mean[time] = WrappedTensorDisposeScope(() => updatedMean[time]
                + smoothingGain.matmul(
                    (smoothedState.Mean[time + 1] - predictedMean[time + 1]).unsqueeze(-1)
                ).squeeze(-1));

            // Smoothed covariance
            smoothedState.Covariance[time] = WrappedTensorDisposeScope(() => updatedCovariance[time] + smoothingGain
                    .matmul(smoothedState.Covariance[time + 1] - predictedCovariance[time + 1])
                    .matmul(smoothingGain.mT)
            );

            var expectationUpdate = outer(smoothedState.Mean[time], smoothedState.Mean[time]) + smoothedState.Covariance[time];
            sxx11[time] = expectationUpdate;
            sxx00[time + 1] = expectationUpdate;
            sxx10[time + 1] = outer(smoothedState.Mean[time + 1], smoothedState.Mean[time]) + smoothedLagOneCovariance;

            if (parameters.OffsetsProvided)
            {
                tx1 += smoothedState.Mean[time];
                tx0 += smoothedState.Mean[time + 1];
                ty1 += observations[time];
                tyx11 += outer(observations[time], smoothedState.Mean[time]);
                tyy11 += outer(observations[time], observations[time]);
            }

            // Compute next smoothing gain for lag one covariance
            if (time > 0)
            {
                smoothingGainNext = WrappedTensorDisposeScope(() => updatedCovariance[time - 1]
                    .matmul(InverseCholesky(parameters.TransitionMatrix.mT, predictedCovariance[time])));

                // Smoothed lag one covariance
                smoothedLagOneCovariance = WrappedTensorDisposeScope(() => updatedCovariance[time]
                    .matmul(smoothingGainNext.mT)
                    + smoothingGain.matmul(smoothedLagOneCovariance
                        - parameters.TransitionMatrix.matmul(updatedCovariance[time]))
                        .matmul(smoothingGainNext.mT));
            }
        }

        var smoothingGain0 = WrappedTensorDisposeScope(() => parameters.InitialCovariance.matmul(
            InverseCholesky(parameters.TransitionMatrix.mT, predictedCovariance[0])
        ));

        // Smoothed initial mean
        var smoothedInitialMean = WrappedTensorDisposeScope(() => parameters.InitialMean + smoothingGain0.matmul(
            (smoothedState.Mean[0] - predictedMean[0]).unsqueeze(-1)
        ).squeeze(-1));

        // Smoothed initial covariance
        var smoothedInitialCovariance = WrappedTensorDisposeScope(() => parameters.InitialCovariance + smoothingGain0
                .matmul(smoothedState.Covariance[0] - predictedCovariance[0])
                .matmul(smoothingGain0.mT));

        // Smoothed lag one covariance at time 0
        smoothedLagOneCovariance = WrappedTensorDisposeScope(() => updatedCovariance[0]
            .matmul(smoothingGain0.mT)
            + smoothingGain.matmul(smoothedLagOneCovariance
                - parameters.TransitionMatrix.matmul(updatedCovariance[0]))
                .matmul(smoothingGain0.mT));

        sxx10[0] = outer(smoothedState.Mean[0], smoothedInitialMean) + smoothedLagOneCovariance;
        sxx00[0] = outer(smoothedInitialMean, smoothedInitialMean) + smoothedInitialCovariance;

        if (parameters.OffsetsProvided)
            tx0 += smoothedInitialMean;

        var state = new SmoothedStateWithAuxiliaryVariables(
            smoothedState: smoothedState,
            smoothedInitialMean: smoothedInitialMean,
            smoothedInitialCovariance: smoothedInitialCovariance
        );

        var stats = parameters.OffsetsProvided ? new SummaryStatisticsEM(
            sxx11: sxx11,
            sxx10: sxx10,
            sxx00: sxx00,
            tx1: tx1,
            tx0: tx0,
            ty1: ty1,
            tyx11: tyx11,
            tyy11: tyy11
        ) : new SummaryStatisticsEM(
            sxx11: sxx11,
            sxx10: sxx10,
            sxx00: sxx00
        );

        return (state, stats);
    }

    public static ExpectationMaximizationResult ExpectationMaximization(
        Tensor observation,
        KalmanFilterParameters parameters,
        int maxIterations = 100,
        double tolerance = 1e-4,
        ParametersToEstimate parametersToEstimate = new())
    {
        var timeBins = observation.size(0);
        var numObservations = (int)observation.size(1);
        var logLikelihood = empty(maxIterations, dtype: ScalarType.Float32, device: parameters.Device);
        var previousLogLikelihood = double.NegativeInfinity;
        var logLikelihoodConst = -0.5 * timeBins * numObservations * Math.Log(2.0 * Math.PI);

        if (parameters.NumObservations != numObservations)
            throw new ArgumentException($"The number of observation dimensions in the parameters ({parameters.NumObservations}) does not match the observations ({numObservations}).");

        var identityStates = eye(parameters.NumStates, dtype: parameters.ScalarType, device: parameters.Device);

        // Precompute constant observation terms reused across EM iterations
        var observationT = observation.mT;
        var autoCorrelationObservations = observationT.matmul(observation);

        using var g = no_grad();

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Filter observations
            var filteredState = Filter(
                timeBins: timeBins,
                observation: observation,
                parameters: parameters);

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
            var (state, statistics) = Smooth(
                filteredState: filteredState,
                observations: observation,
                parameters: parameters);

            // Sufficient statistics
            var sxx00 = statistics.Sxx00.sum([0]);
            var sxx11 = statistics.Sxx11.sum([0]);
            var sxx10 = statistics.Sxx10.sum([0]);

            // Replace einsum with faster matmul
            var crossCorrelationObservations = observationT.matmul(state.SmoothedState.Mean);

            // Update parameters
            if (parametersToEstimate.TransitionMatrix)
            {
                if (parameters.OffsetsProvided)
                {
                    parameters.TransitionMatrix.set_(
                        InverseCholesky(
                            sxx10 - outer(statistics.Tx1, statistics.Tx0) / timeBins,
                            sxx00 - outer(statistics.Tx0, statistics.Tx0) / timeBins
                        )
                    );
                }
                else
                {
                    parameters.TransitionMatrix.set_(
                        InverseCholesky(
                            sxx10,
                            sxx00
                        )
                    );
                }
            }

            if (parametersToEstimate.StateOffset)
            {
                if (parameters.OffsetsProvided)
                {
                    parameters.StateOffset.set_(
                        (statistics.Tx1 - parameters.TransitionMatrix.matmul(statistics.Tx0)) / timeBins
                    );
                }
            }

            if (parametersToEstimate.MeasurementFunction)
            {
                if (parameters.OffsetsProvided)
                {
                    parameters.MeasurementFunction.set_(
                        InverseCholesky(
                            statistics.Tyx11 - outer(statistics.Ty1, statistics.Tx1) / timeBins,
                            sxx11 - outer(statistics.Tx0, statistics.Tx0) / timeBins
                        )
                    );
                }
                else
                {
                    parameters.MeasurementFunction.set_(
                        InverseCholesky(
                            crossCorrelationObservations,
                            sxx11
                        )
                    );
                }
            }

            if (parametersToEstimate.ObservationOffset)
            {
                if (parameters.OffsetsProvided)
                {
                    parameters.ObservationOffset.set_(
                        (statistics.Ty1 - parameters.MeasurementFunction.matmul(statistics.Tx1)) / timeBins
                    );
                }
            }

            if (parametersToEstimate.ProcessNoiseCovariance)
            {
                if (parameters.OffsetsProvided)
                {
                    parameters.ProcessNoiseCovariance.set_(WrappedTensorDisposeScope(() =>
                        EnsureSymmetric(
                            (sxx11 - outer(statistics.Tx1, parameters.StateOffset) - outer(parameters.StateOffset, statistics.Tx1) + timeBins * outer(parameters.StateOffset, parameters.StateOffset) - parameters.TransitionMatrix.matmul(sxx10.mT) - sxx10.matmul(parameters.TransitionMatrix.mT) + linalg.multi_dot([parameters.TransitionMatrix, sxx00, parameters.TransitionMatrix.mT]) + parameters.TransitionMatrix.matmul(outer(statistics.Tx0, parameters.StateOffset)) + outer(parameters.StateOffset, statistics.Tx0).matmul(parameters.TransitionMatrix.mT)) / timeBins
                        )
                    ));
                }
                else
                {
                    parameters.ProcessNoiseCovariance.set_(WrappedTensorDisposeScope(() =>
                        EnsureSymmetric((sxx11 - parameters.TransitionMatrix.matmul(sxx10.mT)) / timeBins)));
                }
            }

            if (parametersToEstimate.MeasurementNoiseCovariance)
            {
                if (parameters.OffsetsProvided)
                {
                    parameters.MeasurementNoiseCovariance.set_(WrappedTensorDisposeScope(() =>
                        EnsureSymmetric(
                            (statistics.Tyy11 - outer(statistics.Ty1, parameters.ObservationOffset) - outer(parameters.ObservationOffset, statistics.Ty1) + timeBins * outer(parameters.ObservationOffset, parameters.ObservationOffset) - parameters.MeasurementFunction.matmul(statistics.Tyx11.mT) - statistics.Tyx11.matmul(parameters.MeasurementFunction.mT) + parameters.MeasurementFunction.matmul(outer(statistics.Tx1, parameters.ObservationOffset)) + linalg.multi_dot([parameters.MeasurementFunction, sxx11, parameters.MeasurementFunction.mT]) + outer(parameters.ObservationOffset, statistics.Tx1).matmul(parameters.MeasurementFunction.mT)) / timeBins
                        )
                    ));
                }
                else
                {
                    var explainedObservationCovariance = parameters.MeasurementFunction.matmul(crossCorrelationObservations.mT);

                    if (parametersToEstimate.MeasurementNoiseCovariance)
                        parameters.MeasurementNoiseCovariance.set_(WrappedTensorDisposeScope(() =>
                            EnsureSymmetric((autoCorrelationObservations - explainedObservationCovariance - explainedObservationCovariance.mT
                                + parameters.MeasurementFunction.matmul(sxx11).matmul(parameters.MeasurementFunction.mT)) / timeBins)));
                }
            }

            if (parametersToEstimate.InitialMean)
                parameters.InitialMean.set_(state.SmoothedInitialMean);

            if (parametersToEstimate.InitialCovariance)
                parameters.InitialCovariance.set_(state.SmoothedInitialCovariance);
        }

        return new ExpectationMaximizationResult(logLikelihood, parameters);
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
        var numObservations = observations.size(1);
        var centered = observations - observations.mean([0], keepdim: true);

        // Build Hankel matrices from observations
        var numCols = (int)(timeBins - 2 * maxLag + 1);

        if (numCols <= 0)
            throw new ArgumentException($"Number of time bins ({timeBins}) must be greater than 2*maxLag ({2 * maxLag}) for subspace identification.");

        var stride = centered.stride();
        var pastView = centered.as_strided([maxLag, numCols, numObservations], [stride[0], stride[0], stride[1]]);
        var past = pastView.permute(0, 2, 1).reshape(maxLag * numObservations, numCols);

        var futureView = centered.narrow(0, maxLag, timeBins - maxLag)
            .as_strided([maxLag, numCols, numObservations], [stride[0], stride[0], stride[1]]);
        var future = futureView.permute(0, 2, 1).reshape(maxLag * numObservations, numCols);

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
        var measurementFunction = observability[TensorIndex.Slice(0, numObservations)];

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
            numStates: (int)effectiveStates,
            numObservations: (int)numObservations,
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

    public LinearDynamicalSystemState OrthogonalizeMeanAndCovariance(LinearDynamicalSystemState state)
    {
        var (_, S, Vt) = linalg.svd(Parameters.MeasurementFunction);
        var SVt = diag(S).matmul(Vt);

        Tensor orthogonalizedMean = null;
        if (state.Mean is not null)
            orthogonalizedMean = matmul(state.Mean, SVt.mT);

        Tensor orthogonalizedCovariance = null;
        if (state.Covariance is not null)
        {
            var auxilary = matmul(SVt, state.Covariance);
            orthogonalizedCovariance = matmul(auxilary, SVt.mT);
        }

        return new LinearDynamicalSystemState(
            orthogonalizedMean,
            orthogonalizedCovariance
        );
    }

    public void UpdateParameters(KalmanFilterParameters updatedParameters)
    {
        if (updatedParameters.TransitionMatrix is not null)
            Parameters.TransitionMatrix.set_(updatedParameters.TransitionMatrix);
        if (updatedParameters.MeasurementFunction is not null)
            Parameters.MeasurementFunction.set_(updatedParameters.MeasurementFunction);
        if (updatedParameters.ProcessNoiseCovariance is not null)
            Parameters.ProcessNoiseCovariance.set_(updatedParameters.ProcessNoiseCovariance);
        if (updatedParameters.MeasurementNoiseCovariance is not null)
            Parameters.MeasurementNoiseCovariance.set_(updatedParameters.MeasurementNoiseCovariance);
        if (updatedParameters.InitialMean is not null)
            Parameters.InitialMean.set_(updatedParameters.InitialMean);
        if (updatedParameters.InitialCovariance is not null)
            Parameters.InitialCovariance.set_(updatedParameters.InitialCovariance);
        if (updatedParameters.StateOffset is not null)
            Parameters.StateOffset.set_(updatedParameters.StateOffset);
        if (updatedParameters.ObservationOffset is not null)
            Parameters.ObservationOffset.set_(updatedParameters.ObservationOffset);
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
