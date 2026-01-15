using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a torch module for convolution operations.
/// </summary>
[XmlInclude(typeof(Container.Sequential))]
[DefaultProperty(nameof(ContainerModule))]
[Combinator]
[Description("Creates a sequential container for torch modules.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ContainerBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    /// <inheritdoc/>
    internal override string BuilderName => "Container";

    /// <summary>
    /// Initializes a new instance of the <see cref="ContainerBuilder"/> class.
    /// </summary>
    public ContainerBuilder()
    {
        Module = new Container.Sequential();
    }

    /// <summary>
    /// Gets or sets the specific container module to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific container module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object ContainerModule
    {
        get => Module;
        set => Module = value;
    }
}
