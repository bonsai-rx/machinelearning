using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates an optimizer.
/// </summary>
[XmlInclude(typeof(Optimizer.Adadelta))]
[XmlInclude(typeof(Optimizer.Adagrad))]
[XmlInclude(typeof(Optimizer.Adam))]
[XmlInclude(typeof(Optimizer.Adamax))]
[XmlInclude(typeof(Optimizer.AdamW))]
[XmlInclude(typeof(Optimizer.AveragedStochasticGradientDescent))]
[XmlInclude(typeof(Optimizer.Lbfgs))]
[XmlInclude(typeof(Optimizer.ResilientBackpropagation))]
[XmlInclude(typeof(Optimizer.RootMeanSquarePropagation))]
[XmlInclude(typeof(Optimizer.StochasticGradientDescent))]
[DefaultProperty(nameof(OptimizerModule))]
[Combinator]
[Description("Creates an optimizer.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class OptimizerBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    string INamedElement.Name => $"Optimizer.{GetElementDisplayName(OptimizerModule)}";

    /// <summary>
    /// Initializes a new instance of the <see cref="OptimizerBuilder"/> class.
    /// </summary>
    public OptimizerBuilder()
    {
        Module = new Optimizer.Adam();
    }
    
    /// <summary>
    /// Gets or sets the specific optimizer to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Optimizer")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific optimizer to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object OptimizerModule
    {
        get => Module;
        set => Module = value;
    }
}
