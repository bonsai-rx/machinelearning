using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Loss;

/// <summary>
/// Represents an operator that creates a cross entropy loss module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"/> for more information.
/// </remarks>
[Description("Creates a cross entropy loss module.")]
[TypeConverter(typeof(TensorOperatorConverter))]
public class CrossEntropy : IScalarTypeProvider
{
    /// <summary>
    /// The manually specified rescaling weight given to each class.
    /// </summary>
    [XmlIgnore]
    [Description("The manually specified rescaling weight given to each class.")]
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
    /// The index to ignore in the target which does not contribute to the input gradient.
    /// </summary>
    [Description("The index to ignore in the target which does not contribute to the input gradient.")]
    public long? IgnoreIndex { get; set; } = null;

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
    /// Creates a cross entropy loss module.
    /// </summary>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.CrossEntropyLoss> Process()
    {
        return Observable.Return(CrossEntropyLoss(Weight, IgnoreIndex, Reduction));
    }

    /// <summary>
    /// Creates a cross entropy loss module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<TorchSharp.Modules.CrossEntropyLoss> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => CrossEntropyLoss(Weight, IgnoreIndex, Reduction));
    }
}
