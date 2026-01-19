using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a padding module.
/// </summary>
[XmlInclude(typeof(Padding.ConstantPad1d))]
[XmlInclude(typeof(Padding.ConstantPad2d))]
[XmlInclude(typeof(Padding.ConstantPad3d))]
[XmlInclude(typeof(Padding.ReflectionPad1d))]
[XmlInclude(typeof(Padding.ReflectionPad2d))]
[XmlInclude(typeof(Padding.ReflectionPad3d))]
[XmlInclude(typeof(Padding.ReplicationPad1d))]
[XmlInclude(typeof(Padding.ReplicationPad2d))]
[XmlInclude(typeof(Padding.ReplicationPad3d))]
[XmlInclude(typeof(Padding.ZeroPad2d))]
[DefaultProperty(nameof(PaddingModule))]
[Combinator]
[Description("Creates a padding module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PaddingModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    internal override string BuilderName => "PaddingModule";

    /// <summary>
    /// Initializes a new instance of the <see cref="PaddingModuleBuilder"/> class.
    /// </summary>
    public PaddingModuleBuilder()
    {
        Module = new Padding.ConstantPad1d();
    }

    /// <summary>
    /// Gets or sets the specific padding module to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific padding module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object PaddingModule
    {
        get => Module;
        set => Module = value;
    }
}
