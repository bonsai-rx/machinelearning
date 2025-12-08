using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a module for distance computations.
/// </summary>
[XmlInclude(typeof(Distance.CosineSimilarity))]
[XmlInclude(typeof(Distance.PairwiseDistance))]
[DefaultProperty(nameof(DistanceModule))]
[Combinator]
[Description("Creates a module for distance computations.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class DistanceModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    /// <summary>
    /// Initializes a new instance of the <see cref="DistanceModuleBuilder"/> class.
    /// </summary>
    public DistanceModuleBuilder()
    {
        Module = new Distance.CosineSimilarity();
    }
    
    /// <summary>
    /// Gets or sets the specific distance module to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific distance module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object DistanceModule
    {
        get => Module;
        set => Module = value;
    }
}
