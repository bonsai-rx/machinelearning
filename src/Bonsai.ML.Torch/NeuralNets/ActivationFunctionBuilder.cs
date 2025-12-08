using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates an activation function.
/// </summary>
[XmlInclude(typeof(NonLinearActivations.ContinuouslyDifferentiableExponential))]
[XmlInclude(typeof(NonLinearActivations.Exponential))]
[XmlInclude(typeof(NonLinearActivations.Gated))]
[XmlInclude(typeof(NonLinearActivations.GaussianError))]
[XmlInclude(typeof(NonLinearActivations.HardShrinkage))]
[XmlInclude(typeof(NonLinearActivations.HardSigmoid))]
[XmlInclude(typeof(NonLinearActivations.Hardswish))]
[XmlInclude(typeof(NonLinearActivations.HardTanh))]
[XmlInclude(typeof(NonLinearActivations.LeakyRectified))]
[XmlInclude(typeof(NonLinearActivations.LogSigmoid))]
[XmlInclude(typeof(NonLinearActivations.LogSoftmax))]
[XmlInclude(typeof(NonLinearActivations.Mish))]
[XmlInclude(typeof(NonLinearActivations.MultiheadAttention))]
[XmlInclude(typeof(NonLinearActivations.ParametricRectified))]
[XmlInclude(typeof(NonLinearActivations.RandomizedLeakyRectified))]
[XmlInclude(typeof(NonLinearActivations.Rectified))]
[XmlInclude(typeof(NonLinearActivations.RectifiedBounded))]
[XmlInclude(typeof(NonLinearActivations.ScaledExponential))]
[XmlInclude(typeof(NonLinearActivations.Sigmoid))]
[XmlInclude(typeof(NonLinearActivations.SigmoidWeighted))]
[XmlInclude(typeof(NonLinearActivations.Softmax))]
[XmlInclude(typeof(NonLinearActivations.Softmax2d))]
[XmlInclude(typeof(NonLinearActivations.Softmin))]
[XmlInclude(typeof(NonLinearActivations.Softplus))]
[XmlInclude(typeof(NonLinearActivations.SoftShrinkage))]
[XmlInclude(typeof(NonLinearActivations.Softsign))]
[XmlInclude(typeof(NonLinearActivations.Tanh))]
[XmlInclude(typeof(NonLinearActivations.TanhShrinkage))]
[XmlInclude(typeof(NonLinearActivations.Threshold))]
[DefaultProperty(nameof(ActivationFunction))]
[Combinator]
[Description("Creates an activation function.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ActivationFunctionBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    string INamedElement.Name => $"ActivationFunction.{GetElementDisplayName(ActivationFunction)}";

    /// <summary>
    /// Initializes a new instance of the <see cref="ActivationFunctionBuilder"/> class.
    /// </summary>
    public ActivationFunctionBuilder()
    {
        Module = new NonLinearActivations.Rectified();
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
