using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a transformer module.
/// </summary>
[XmlInclude(typeof(Transformer.Transformer))]
[XmlInclude(typeof(Transformer.TransformerDecoder))]
[XmlInclude(typeof(Transformer.TransformerDecoderLayer))]
[XmlInclude(typeof(Transformer.TransformerEncoder))]
[XmlInclude(typeof(Transformer.TransformerEncoderLayer))]
[DefaultProperty(nameof(TransformerModule))]
[Combinator]
[Description("Creates a transformer module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class TransformerModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    internal override string BuilderName => "TransformerModule";

    /// <summary>
    /// Initializes a new instance of the <see cref="TransformerModuleBuilder"/> class.
    /// </summary>
    public TransformerModuleBuilder()
    {
        Module = new Transformer.Transformer();
    }

    /// <summary>
    /// Gets or sets the specific transformer module to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific transformer module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object TransformerModule
    {
        get => Module;
        set => Module = value;
    }
}
