using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates an activation function.
/// </summary>
[XmlInclude(typeof(ActivationFunction.CELU))]
[XmlInclude(typeof(ActivationFunction.ELU))]
[XmlInclude(typeof(ActivationFunction.GLU))]
[XmlInclude(typeof(ActivationFunction.GELU))]
[XmlInclude(typeof(ActivationFunction.Hardshrink))]
[XmlInclude(typeof(ActivationFunction.Hardsigmoid))]
[XmlInclude(typeof(ActivationFunction.Hardswish))]
[XmlInclude(typeof(ActivationFunction.Hardtanh))]
[XmlInclude(typeof(ActivationFunction.LeakyReLU))]
[XmlInclude(typeof(ActivationFunction.LogSigmoid))]
[XmlInclude(typeof(ActivationFunction.LogSoftmax))]
[XmlInclude(typeof(ActivationFunction.Mish))]
[XmlInclude(typeof(ActivationFunction.MultiheadAttention))]
[XmlInclude(typeof(ActivationFunction.PReLU))]
[XmlInclude(typeof(ActivationFunction.RReLU))]
[XmlInclude(typeof(ActivationFunction.Rectified))]
[XmlInclude(typeof(ActivationFunction.RectifiedBounded))]
[XmlInclude(typeof(ActivationFunction.SELU))]
[XmlInclude(typeof(ActivationFunction.Sigmoid))]
[XmlInclude(typeof(ActivationFunction.SiLU))]
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
        Module = new ActivationFunction.Rectified();
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
