using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Module;

/// <summary>
/// Creates a TransformerEncoder module.
/// </summary>
[Combinator]
[Description("Creates a TransformerEncoder module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class TransformerEncoder
{
    /// <summary>
    /// The encoder_layer parameter for the TransformerEncoder module.
    /// </summary>
    [Description("The encoder_layer parameter for the TransformerEncoder module")]
    [XmlIgnore]
    public TorchSharp.Modules.TransformerEncoderLayer EncoderLayer { get; set; }

    /// <summary>
    /// The num_layers parameter for the TransformerEncoder module.
    /// </summary>
    [Description("The num_layers parameter for the TransformerEncoder module")]
    public long NumLayers { get; set; }

    /// <summary>
    /// Generates an observable sequence that creates a TransformerEncoderModule module.
    /// </summary>
    public IObservable<IModule<Tensor, Tensor, Tensor, Tensor>> Process()
    {
        return Observable.Return(TransformerEncoder(EncoderLayer, NumLayers));
    }
}
