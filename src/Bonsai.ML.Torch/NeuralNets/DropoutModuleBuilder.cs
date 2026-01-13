using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a module for dropout computations.
/// </summary>
[XmlInclude(typeof(Dropout.AlphaDropout))]
[XmlInclude(typeof(Dropout.Dropout))]
[XmlInclude(typeof(Dropout.Dropout1d))]
[XmlInclude(typeof(Dropout.Dropout2d))]
[XmlInclude(typeof(Dropout.Dropout3d))]
[XmlInclude(typeof(Dropout.FeatureAlphaDropout))]
[DefaultProperty(nameof(DropoutModule))]
[Combinator]
[Description("Creates a module for dropout computations.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class DropoutModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    /// <summary>
    /// Initializes a new instance of the <see cref="DropoutModuleBuilder"/> class.
    /// </summary>
    public DropoutModuleBuilder()
    {
        Module = new Dropout.AlphaDropout();
    }
    
    /// <summary>
    /// Gets or sets the specific dropout module to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific dropout module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object DropoutModule
    {
        get => Module;
        set => Module = value;
    }
}
