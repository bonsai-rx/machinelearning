using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a normalization module.
/// </summary>
[XmlInclude(typeof(Normalization.BatchNorm1d))]
[XmlInclude(typeof(Normalization.BatchNorm2d))]
[XmlInclude(typeof(Normalization.BatchNorm3d))]
[XmlInclude(typeof(Normalization.GroupNorm))]
[XmlInclude(typeof(Normalization.InstanceNorm1d))]
[XmlInclude(typeof(Normalization.InstanceNorm2d))]
[XmlInclude(typeof(Normalization.InstanceNorm3d))]
[XmlInclude(typeof(Normalization.LayerNorm))]
[XmlInclude(typeof(Normalization.LocalResponseNorm))]
[DefaultProperty(nameof(NormalizationModule))]
[Combinator]
[Description("Creates a normalization module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class NormalizationModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    /// <summary>
    /// Initializes a new instance of the <see cref="NormalizationModuleBuilder"/> class.
    /// </summary>
    public NormalizationModuleBuilder()
    {
        Module = new Normalization.BatchNorm1d();
    }
    
    /// <summary>
    /// Gets or sets the specific normalization module to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific normalization module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object NormalizationModule
    {
        get => Module;
        set => Module = value;
    }
}
