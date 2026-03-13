using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Represents an operator that creates a multi-label soft margin loss (MultiLabelSoftMarginLoss) module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html"/> for more information.
/// </remarks>
[Description("Creates a multi-label soft margin loss (MultiLabelSoftMarginLoss) module.")]
[TypeConverter(typeof(TensorOperatorConverter))]
public class MultiLabelSoftMargin : IScalarTypeProvider
{
    /// <summary>
    /// The manual rescaling weight given to each class.
    /// </summary>
    [XmlIgnore]
    [Description("The manual rescaling weight given to each class.")]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor Weight { get; set; } = null;

    /// <summary>
    /// The values of the weight tensor in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(Weight))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string WeightXml
    {
        get => TensorConverter.ConvertToString(Weight, Type);
        set => Weight = TensorConverter.ConvertFromString(value, Type);
    }

    /// <summary>
    /// The reduction type to apply to the output.
    /// </summary>
    [Description("The reduction type to apply to the output.")]
    public Reduction Reduction { get; set; } = Reduction.Mean;

    /// <summary>
    /// The data type of the tensor elements.
    /// </summary>
    [Description("The data type of the tensor elements.")]
    public ScalarType Type { get; set; } = ScalarType.Float32;

    /// <summary>
    /// Creates a multi-label soft margin loss (MultiLabelSoftMarginLoss) module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.MultiLabelSoftMarginLoss> Process()
    {
        return Observable.Return(MultiLabelSoftMarginLoss(Weight, Reduction));
    }

    /// <summary>
    /// Creates a multi-label soft margin loss (MultiLabelSoftMarginLoss) module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.MultiLabelSoftMarginLoss> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => MultiLabelSoftMarginLoss(Weight, Reduction));
    }
}
