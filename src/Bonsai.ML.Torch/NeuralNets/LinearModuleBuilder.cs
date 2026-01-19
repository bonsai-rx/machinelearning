using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a linear module.
/// </summary>
[XmlInclude(typeof(Linear.Bilinear))]
[XmlInclude(typeof(Linear.Identity))]
[XmlInclude(typeof(Linear.Linear))]
[DefaultProperty(nameof(LinearModule))]
[Combinator]
[Description("Creates a linear module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LinearModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    internal override string BuilderName => "LinearModule";

    /// <summary>
    /// Initializes a new instance of the <see cref="LinearModuleBuilder"/> class.
    /// </summary>
    public LinearModuleBuilder()
    {
        Module = new Linear.Linear();
    }

    /// <summary>
    /// Gets or sets the specific linear module.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific linear module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object LinearModule
    {
        get => Module;
        set => Module = value;
    }
}
