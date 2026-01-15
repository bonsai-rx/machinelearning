using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a recurrent neural network module.
/// </summary>
[XmlInclude(typeof(Recurrent.GatedRecurrentUnit))]
[XmlInclude(typeof(Recurrent.GatedRecurrentUnitCell))]
[XmlInclude(typeof(Recurrent.LongShortTermMemory))]
[XmlInclude(typeof(Recurrent.LongShortTermMemoryCell))]
[XmlInclude(typeof(Recurrent.RecurrentNeuralNetwork))]
[XmlInclude(typeof(Recurrent.RecurrentNeuralNetworkCell))]
[DefaultProperty(nameof(RecurrentModule))]
[Combinator]
[Description("Creates a recurrent neural network module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class RecurrentModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    internal override string BuilderName => "RecurrentModule";

    /// <summary>
    /// Initializes a new instance of the <see cref="RecurrentModuleBuilder"/> class.
    /// </summary>
    public RecurrentModuleBuilder()
    {
        Module = new Recurrent.RecurrentNeuralNetwork();
    }

    /// <summary>
    /// Gets or sets the specific recurrent neural network module to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific recurrent neural network module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object RecurrentModule
    {
        get => Module;
        set => Module = value;
    }
}
