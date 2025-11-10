using System;
using System.ComponentModel;
using System.Reactive.Linq;
using PointProcessDecoder.Core;
using PointProcessDecoder.Core.Decoder;
using static TorchSharp.torch;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Transforms the output of a decoder model into a <see cref="DecoderData"/> object.
/// </summary>
[Combinator]
[WorkflowElementCategory(ElementCategory.Transform)]
[Description("Transforms the output of a decoder model into a DecoderData object.")]
public class GetDecoderData : IPointProcessModelReference
{
    /// <summary>
    /// The name of the point process model to use.
    /// </summary>
    [TypeConverter(typeof(PointProcessModelNameConverter))]
    [Description("The name of the point process model to use.")]
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Converts the input tensor representing the decoded output of a point process decoder model into a packaged <see cref="DecoderData"/> object.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<DecoderDataFrame> Process(IObservable<Tensor> source)
    {
        var modelName = Name;
        return source.Select(input =>
        {
            var model = PointProcessModelManager.GetModel(modelName);
            var decoderData = new DecoderData(model.StateSpace, input);
            return new DecoderDataFrame(
                decoderData,
                modelName);
        });
    }
}

public readonly struct DecoderDataFrame(
    DecoderData decoderData,
    string name) : IPointProcessModelReference
{
    public DecoderData DecoderData => decoderData;
    public string Name => name;
}