using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a module for pooling operations.
/// </summary>
[XmlInclude(typeof(Pooling.AdaptiveAvgPool1d))]
[XmlInclude(typeof(Pooling.AdaptiveAvgPool2d))]
[XmlInclude(typeof(Pooling.AdaptiveAvgPool3d))]
[XmlInclude(typeof(Pooling.AdaptiveMaxPool1d))]
[XmlInclude(typeof(Pooling.AdaptiveMaxPool2d))]
[XmlInclude(typeof(Pooling.AdaptiveMaxPool3d))]
[XmlInclude(typeof(Pooling.AvgPool1d))]
[XmlInclude(typeof(Pooling.AvgPool2d))]
[XmlInclude(typeof(Pooling.AvgPool3d))]
[XmlInclude(typeof(Pooling.FractionalMaxPool2d))]
[XmlInclude(typeof(Pooling.FractionalMaxPool3d))]
[XmlInclude(typeof(Pooling.LPPool1d))]
[XmlInclude(typeof(Pooling.LPPool2d))]
[XmlInclude(typeof(Pooling.MaxPool1d))]
[XmlInclude(typeof(Pooling.MaxPool2d))]
[XmlInclude(typeof(Pooling.MaxPool3d))]
[XmlInclude(typeof(Pooling.MaxUnpool1d))]
[XmlInclude(typeof(Pooling.MaxUnpool2d))]
[XmlInclude(typeof(Pooling.MaxUnpool3d))]
[DefaultProperty(nameof(PoolingModule))]
[Combinator]
[Description("Creates a module for pooling operations.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PoolingModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    internal override string BuilderName => "PoolingModule";

    /// <summary>
    /// Initializes a new instance of the <see cref="PoolingModuleBuilder"/> class.
    /// </summary>
    public PoolingModuleBuilder()
    {
        Module = new Pooling.AdaptiveAvgPool1d();
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
