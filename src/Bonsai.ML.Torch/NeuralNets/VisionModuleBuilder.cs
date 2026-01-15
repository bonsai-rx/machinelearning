using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a module for image processing.
/// </summary>
[XmlInclude(typeof(Vision.PixelShuffle))]
[XmlInclude(typeof(Vision.PixelUnshuffle))]
[XmlInclude(typeof(Vision.Upsample))]
[DefaultProperty(nameof(VisionModule))]
[Combinator]
[Description("Creates a module for image processing.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class VisionModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    internal override string BuilderName => "VisionModule";

    /// <summary>
    /// Initializes a new instance of the <see cref="VisionModuleBuilder"/> class.
    /// </summary>
    public VisionModuleBuilder()
    {
        Module = new Vision.PixelShuffle();
    }

    /// <summary>
    /// Gets or sets the specific vision module to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific vision module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object VisionModule
    {
        get => Module;
        set => Module = value;
    }
}
