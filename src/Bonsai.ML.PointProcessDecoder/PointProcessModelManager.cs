using System;
using System.Collections.Generic;
using System.Reactive.Disposables;

using static TorchSharp.torch;

using PointProcessDecoder.Core;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.StateSpace;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.Decoder;
using PointProcessDecoder.Core.Likelihood;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Manages the point process models.
/// </summary>
public static class PointProcessModelManager
{
    private static readonly Dictionary<string, PointProcessModel> models = [];

    public static PointProcessModel GetModel(string name)
    {
        return models.TryGetValue(name, out var model) ? model : throw new InvalidOperationException($"Model with name {name} not found.");
    }

    internal static PointProcessModelDisposable Reserve(
        string name,
        EstimationMethod estimationMethod,
        TransitionsType transitionsType,
        EncoderType encoderType,
        DecoderType decoderType,
        StateSpaceType stateSpaceType,
        LikelihoodType likelihoodType,
        double[] minStateSpace,
        double[] maxStateSpace,
        long[] stepsStateSpace,
        double[] observationBandwidth,
        int stateSpaceDimensions,
        int? markDimensions = null,
        int? markChannels = null,
        double[]? markBandwidth = null,
        bool ignoreNoSpikes = false,
        int? nUnits = null,
        double? distanceThreshold = null,
        double? sigmaRandomWalk = null,
        int? kernelLimit = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        var model = new PointProcessModel(
            estimationMethod: estimationMethod,
            transitionsType: transitionsType,
            encoderType: encoderType,
            decoderType: decoderType,
            stateSpaceType: stateSpaceType,
            likelihoodType: likelihoodType,
            minStateSpace: minStateSpace,
            maxStateSpace: maxStateSpace,
            stepsStateSpace: stepsStateSpace,
            observationBandwidth: observationBandwidth,
            stateSpaceDimensions: stateSpaceDimensions,
            markDimensions: markDimensions,
            markChannels: markChannels,
            markBandwidth: markBandwidth,
            ignoreNoSpikes: ignoreNoSpikes,
            nUnits: nUnits,
            distanceThreshold: distanceThreshold,
            sigmaRandomWalk: sigmaRandomWalk,
            kernelLimit: kernelLimit,
            device: device,
            scalarType: scalarType
        );

        models.Add(name, model);
        
        return new PointProcessModelDisposable(
            model, 
            Disposable.Create(() => {
                models.Remove(name);
            })
        );
    }

    internal static PointProcessModelDisposable Load(
        string name,
        string path,
        Device? device = null
    )
    {
        var model = PointProcessModel.Load(path, device) as PointProcessModel ?? throw new InvalidOperationException("The model could not be loaded.");
        models.Add(name, model);
        
        return new PointProcessModelDisposable(
            model, 
            Disposable.Create(() => {
                models.Remove(name);
            })
        );
    }
}