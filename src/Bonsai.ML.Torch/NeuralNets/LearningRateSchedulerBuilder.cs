using System.ComponentModel;
using System.Xml.Serialization;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a learning rate scheduler.
/// </summary>
[XmlInclude(typeof(LearningRateScheduler.Constant))]
[XmlInclude(typeof(LearningRateScheduler.CosineAnnealing))]
[XmlInclude(typeof(LearningRateScheduler.Exponential))]
[XmlInclude(typeof(LearningRateScheduler.Linear))]
[XmlInclude(typeof(LearningRateScheduler.MultiStep))]
[XmlInclude(typeof(LearningRateScheduler.OneCycle))]
[XmlInclude(typeof(LearningRateScheduler.Polynomial))]
[XmlInclude(typeof(LearningRateScheduler.ReduceOnPlateau))]
[XmlInclude(typeof(LearningRateScheduler.Step))]
[DefaultProperty(nameof(LearningRateScheduler))]
[Combinator]
[Description("Creates a learning rate scheduler.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class LearningRateSchedulerBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    internal override string BuilderName => "LearningRateScheduler";

    /// <summary>
    /// Initializes a new instance of the <see cref="LearningRateSchedulerBuilder"/> class.
    /// </summary>
    public LearningRateSchedulerBuilder()
    {
        Module = new LearningRateScheduler.Constant();
    }

    /// <summary>
    /// Gets or sets the specific learning rate scheduler to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Scheduler")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific learning rate scheduler to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object LearningRateScheduler
    {
        get => Module;
        set => Module = value;
    }
}
