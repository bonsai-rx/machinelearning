using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a padding module.
/// </summary>
[XmlInclude(typeof(Padding.ConstantPad1D))]
[XmlInclude(typeof(Padding.ConstantPad2D))]
[XmlInclude(typeof(Padding.ConstantPad3D))]
[XmlInclude(typeof(Padding.ReflectionPad1D))]
[XmlInclude(typeof(Padding.ReflectionPad2D))]
[XmlInclude(typeof(Padding.ReflectionPad3D))]
[XmlInclude(typeof(Padding.ReplicationPad1D))]
[XmlInclude(typeof(Padding.ReplicationPad2D))]
[XmlInclude(typeof(Padding.ReplicationPad3D))]
[XmlInclude(typeof(Padding.ZeroPad2D))]
[DefaultProperty(nameof(PaddingModule))]
[Description("Creates a padding module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PaddingModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    internal override string BuilderName => "PaddingModule";

    /// <summary>
    /// Initializes a new instance of the <see cref="PaddingModuleBuilder"/> class.
    /// </summary>
    public PaddingModuleBuilder()
    {
        Module = new Padding.ConstantPad1D();
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
