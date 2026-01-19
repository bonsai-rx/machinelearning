using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a loss module.
/// </summary>
[XmlInclude(typeof(Loss.BinaryCrossEntropy))]
[XmlInclude(typeof(Loss.BinaryCrossEntropyWithLogits))]
[XmlInclude(typeof(Loss.ConnectionistTemporalClassification))]
[XmlInclude(typeof(Loss.CosineEmbedding))]
[XmlInclude(typeof(Loss.CrossEntropy))]
[XmlInclude(typeof(Loss.GaussianNegativeLogLikelihood))]
[XmlInclude(typeof(Loss.HingeEmbedding))]
[XmlInclude(typeof(Loss.Huber))]
[XmlInclude(typeof(Loss.KullbackLeiblerDivergence))]
[XmlInclude(typeof(Loss.MarginRanking))]
[XmlInclude(typeof(Loss.MeanAbsoluteError))]
[XmlInclude(typeof(Loss.MeanSquaredError))]
[XmlInclude(typeof(Loss.MultiMargin))]
[XmlInclude(typeof(Loss.MultiLabelMargin))]
[XmlInclude(typeof(Loss.MultiLabelSoftMargin))]
[XmlInclude(typeof(Loss.NegativeLogLikelihood))]
[XmlInclude(typeof(Loss.PoissonNegativeLogLikelihood))]
[XmlInclude(typeof(Loss.SmoothL1))]
[XmlInclude(typeof(Loss.SoftMargin))]
[XmlInclude(typeof(Loss.TripletMargin))]
[DefaultProperty(nameof(LossModule))]
[Combinator]
[Description("Creates a loss module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LossBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    internal override string BuilderName => "Loss";

    /// <summary>
    /// Initializes a new instance of the <see cref="LossBuilder"/> class.
    /// </summary>
    public LossBuilder()
    {
        Module = new Loss.GaussianNegativeLogLikelihood();
    }

    /// <summary>
    /// Gets or sets the specific loss module.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Loss")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific loss module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object LossModule
    {
        get => Module;
        set => Module = value;
    }
}
