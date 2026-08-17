using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a torch module for convolution operations.
/// </summary>
[XmlInclude(typeof(Convolution.Conv1D))]
[XmlInclude(typeof(Convolution.Conv2D))]
[XmlInclude(typeof(Convolution.Conv3D))]
[XmlInclude(typeof(Convolution.ConvTranspose1D))]
[XmlInclude(typeof(Convolution.ConvTranspose2D))]
[XmlInclude(typeof(Convolution.ConvTranspose3D))]
[XmlInclude(typeof(Convolution.Fold))]
[XmlInclude(typeof(Convolution.Unfold))]
[DefaultProperty(nameof(ConvolutionModule))]
[Description("Creates a torch module for convolution operations.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ConvolutionModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    internal override string BuilderName => "ConvolutionModule";

    /// <summary>
    /// Initializes a new instance of the <see cref="ConvolutionModuleBuilder"/> class.
    /// </summary>
    public ConvolutionModuleBuilder()
    {
        Module = new Convolution.Conv1D();
    }

    /// <summary>
    /// Gets or sets the specific convolution module to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific convolution module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object ConvolutionModule
    {
        get => Module;
        set => Module = value;
    }
}
