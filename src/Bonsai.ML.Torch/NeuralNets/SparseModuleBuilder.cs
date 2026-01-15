using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a module for sparsification.
/// </summary>
[XmlInclude(typeof(Sparse.Embedding))]
[XmlInclude(typeof(Sparse.EmbeddingBag))]
[XmlInclude(typeof(Sparse.EmbeddingBagFromPretrained))]
[XmlInclude(typeof(Sparse.EmbeddingFromPretrained))]
[DefaultProperty(nameof(SparseModule))]
[Combinator]
[Description("Creates a module for sparsification.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SparseModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    internal override string BuilderName => "SparseModule";

    /// <summary>
    /// Initializes a new instance of the <see cref="SparseModuleBuilder"/> class.
    /// </summary>
    public SparseModuleBuilder()
    {
        Module = new Sparse.Embedding();
    }

    /// <summary>
    /// Gets or sets the specific sparse module to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific sparse module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object SparseModule
    {
        get => Module;
        set => Module = value;
    }
}
