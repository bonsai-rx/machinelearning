using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a module for flattening tensors.
/// </summary>
[XmlInclude(typeof(Flatten.Flatten))]
[XmlInclude(typeof(Flatten.Unflatten))]
[DefaultProperty(nameof(FlattenModule))]
[Combinator]
[Description("Creates a module for flattening tensors.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class FlattenModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    internal override string BuilderName => "FlattenModule";

    /// <summary>
    /// Initializes a new instance of the <see cref="FlattenModuleBuilder"/> class.
    /// </summary>
    public FlattenModuleBuilder()
    {
        Module = new Flatten.Flatten();
    }

    /// <summary>
    /// Gets or sets the specific flatten module to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific flatten module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object FlattenModule
    {
        get => Module;
        set => Module = value;
    }
}
