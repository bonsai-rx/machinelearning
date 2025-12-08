using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a module for shuffling input.
/// </summary>
[XmlInclude(typeof(Shuffle.ChannelShuffle))]
[DefaultProperty(nameof(ShuffleModule))]
[Combinator]
[Description("Creates a module for shuffling input.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ShuffleModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    /// <summary>
    /// Initializes a new instance of the <see cref="ShuffleModuleBuilder"/> class.
    /// </summary>
    public ShuffleModuleBuilder()
    {
        Module = new Shuffle.ChannelShuffle();
    }
    
    /// <summary>
    /// Gets or sets the specific shuffle module to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific shuffle module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object ShuffleModule
    {
        get => Module;
        set => Module = value;
    }
}
