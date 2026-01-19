using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates an activation function.
/// </summary>
[XmlInclude(typeof(ActivationFunction.Celu))]
[XmlInclude(typeof(ActivationFunction.Elu))]
[XmlInclude(typeof(ActivationFunction.Glu))]
[XmlInclude(typeof(ActivationFunction.Gelu))]
[XmlInclude(typeof(ActivationFunction.Hardshrink))]
[XmlInclude(typeof(ActivationFunction.Hardsigmoid))]
[XmlInclude(typeof(ActivationFunction.Hardswish))]
[XmlInclude(typeof(ActivationFunction.Hardtanh))]
[XmlInclude(typeof(ActivationFunction.LeakyRelu))]
[XmlInclude(typeof(ActivationFunction.LogSigmoid))]
[XmlInclude(typeof(ActivationFunction.LogSoftmax))]
[XmlInclude(typeof(ActivationFunction.Mish))]
[XmlInclude(typeof(ActivationFunction.MultiheadAttention))]
[XmlInclude(typeof(ActivationFunction.Prelu))]
[XmlInclude(typeof(ActivationFunction.Rrelu))]
[XmlInclude(typeof(ActivationFunction.Relu))]
[XmlInclude(typeof(ActivationFunction.ReluBounded))]
[XmlInclude(typeof(ActivationFunction.Selu))]
[XmlInclude(typeof(ActivationFunction.Sigmoid))]
[XmlInclude(typeof(ActivationFunction.Silu))]
[XmlInclude(typeof(ActivationFunction.Softmax))]
[XmlInclude(typeof(ActivationFunction.Softmax2d))]
[XmlInclude(typeof(ActivationFunction.Softmin))]
[XmlInclude(typeof(ActivationFunction.Softplus))]
[XmlInclude(typeof(ActivationFunction.Softshrink))]
[XmlInclude(typeof(ActivationFunction.Softsign))]
[XmlInclude(typeof(ActivationFunction.Tanh))]
[XmlInclude(typeof(ActivationFunction.Tanhshrink))]
[XmlInclude(typeof(ActivationFunction.Threshold))]
[DefaultProperty(nameof(ActivationFunction))]
[Combinator]
[Description("Creates an activation function.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ActivationFunctionBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    internal override string BuilderName => "ActivationFunction";

    /// <summary>
    /// Initializes a new instance of the <see cref="ActivationFunctionBuilder"/> class.
    /// </summary>
    public ActivationFunctionBuilder()
    {
        Module = new ActivationFunction.Relu();
    }

    /// <summary>
    /// Gets or sets the specific activation function to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific activation function to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object ActivationFunction
    {
        get => Module;
        set => Module = value;
    }
}
