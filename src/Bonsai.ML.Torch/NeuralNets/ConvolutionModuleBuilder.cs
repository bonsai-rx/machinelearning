using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a torch module for convolution operations.
/// </summary>
[XmlInclude(typeof(Convolution.Conv1d))]
[XmlInclude(typeof(Convolution.Conv2d))]
[XmlInclude(typeof(Convolution.Conv3d))]
[XmlInclude(typeof(Convolution.ConvTranspose1d))]
[XmlInclude(typeof(Convolution.ConvTranspose2d))]
[XmlInclude(typeof(Convolution.ConvTranspose3d))]
[XmlInclude(typeof(Convolution.Fold))]
[XmlInclude(typeof(Convolution.Unfold))]
[DefaultProperty(nameof(ConvolutionModule))]
[Combinator]
[Description("Creates a torch module for convolution operations.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ConvolutionModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    /// <summary>
    /// Initializes a new instance of the <see cref="ConvolutionModuleBuilder"/> class.
    /// </summary>
    public ConvolutionModuleBuilder()
    {
        Module = new Convolution.Conv1d();
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
