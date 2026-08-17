using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a module for pooling operations.
/// </summary>
[XmlInclude(typeof(Pooling.AdaptiveAvgPool1D))]
[XmlInclude(typeof(Pooling.AdaptiveAvgPool2D))]
[XmlInclude(typeof(Pooling.AdaptiveAvgPool3D))]
[XmlInclude(typeof(Pooling.AdaptiveMaxPool1D))]
[XmlInclude(typeof(Pooling.AdaptiveMaxPool2D))]
[XmlInclude(typeof(Pooling.AdaptiveMaxPool3D))]
[XmlInclude(typeof(Pooling.AvgPool1D))]
[XmlInclude(typeof(Pooling.AvgPool2D))]
[XmlInclude(typeof(Pooling.AvgPool3D))]
[XmlInclude(typeof(Pooling.FractionalMaxPool2D))]
[XmlInclude(typeof(Pooling.FractionalMaxPool3D))]
[XmlInclude(typeof(Pooling.LPPool1D))]
[XmlInclude(typeof(Pooling.LPPool2D))]
[XmlInclude(typeof(Pooling.MaxPool1D))]
[XmlInclude(typeof(Pooling.MaxPool2D))]
[XmlInclude(typeof(Pooling.MaxPool3D))]
[XmlInclude(typeof(Pooling.MaxUnpool1D))]
[XmlInclude(typeof(Pooling.MaxUnpool2D))]
[XmlInclude(typeof(Pooling.MaxUnpool3D))]
[DefaultProperty(nameof(PoolingModule))]
[Description("Creates a module for pooling operations.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PoolingModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    internal override string BuilderName => "PoolingModule";

    /// <summary>
    /// Initializes a new instance of the <see cref="PoolingModuleBuilder"/> class.
    /// </summary>
    public PoolingModuleBuilder()
    {
        Module = new Pooling.AdaptiveAvgPool1D();
    }

    /// <summary>
    /// Gets or sets the specific pooling module to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific pooling module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object PoolingModule
    {
        get => Module;
        set => Module = value;
    }
}
