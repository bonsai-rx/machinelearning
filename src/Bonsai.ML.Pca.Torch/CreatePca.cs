using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reactive.Linq;
using System.Reflection;
using System.Xml.Serialization;
using Bonsai.Expressions;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Represents an operator that creates a PCA model.
/// </summary>
[XmlInclude(typeof(Pca))]
[XmlInclude(typeof(ProbabilisticPca))]
[XmlInclude(typeof(OnlineProbabilisticPca))]
[XmlInclude(typeof(OnlinePcaGha))]
[WorkflowElementCategory(ElementCategory.Source)]
[Description("Creates a PCA model.")]
public class CreatePca : ZeroArgumentExpressionBuilder, INamedElement, ICustomTypeDescriptor
{
    string INamedElement.Name => $"CreatePca.{ModelType}";

    /// <summary>
    /// Gets or sets the PCA model to create.
    /// </summary>
    [Browsable(false)]
    public PcaBaseModel Model { get; set; } = new Pca();

    /// <summary>
    /// Gets or sets the type of PCA model to create.
    /// </summary>
    [XmlIgnore]
    [RefreshProperties(RefreshProperties.All)]
    [Description("The type of PCA model to create.")]
    [Category("Combinator")]
    public PcaModelType ModelType
    {
        get => Model switch
        {
            ProbabilisticPca => PcaModelType.ProbabilisticPca,
            OnlineProbabilisticPca => PcaModelType.OnlineProbabilisticPca,
            OnlinePcaGha => PcaModelType.OnlinePcaGha,
            _ => PcaModelType.Pca,
        };
        set
        {
            if (value == ModelType) return;
            Model = value switch
            {
                PcaModelType.ProbabilisticPca => new ProbabilisticPca(),
                PcaModelType.OnlineProbabilisticPca => new OnlineProbabilisticPca(),
                PcaModelType.OnlinePcaGha => new OnlinePcaGha(),
                _ => new Pca(),
            };
        }
    }

    /// <inheritdoc/>
    public override Expression Build(IEnumerable<Expression> arguments)
    {
        var processMethod = typeof(CreatePca).GetMethod(
            nameof(Process),
            BindingFlags.Static | BindingFlags.NonPublic);

        var modelType = Model.GetType();
        var genericMethod = processMethod.MakeGenericMethod(modelType);

        return Expression.Call(genericMethod, Expression.Constant(Model));
    }

    static IObservable<T> Process<T>(T model) where T : PcaBaseModel
    {
        return Observable.Using(
            () => model,
            m => Observable.Return(m)
                .Concat(Observable.Never(m))
                .Finally(() => m.Dispose())
        );
    }

    PropertyDescriptorCollection ICustomTypeDescriptor.GetProperties(Attribute[]? attributes)
    {
        var properties = TypeDescriptor.GetProperties(this, attributes, true).Cast<PropertyDescriptor>();
        var modelProperties = TypeDescriptor.GetProperties(Model, attributes)
            .Cast<PropertyDescriptor>()
            .Where(descriptor => !descriptor.IsReadOnly);
        return new PropertyDescriptorCollection(properties.Concat(modelProperties).ToArray());
    }

    PropertyDescriptorCollection ICustomTypeDescriptor.GetProperties() =>
        ((ICustomTypeDescriptor)this).GetProperties(Array.Empty<Attribute>());

    object ICustomTypeDescriptor.GetPropertyOwner(PropertyDescriptor? pd) =>
        pd == null || pd.ComponentType.IsInstanceOfType(Model) ? Model : this;

    AttributeCollection ICustomTypeDescriptor.GetAttributes() => TypeDescriptor.GetAttributes(this, true);

    string? ICustomTypeDescriptor.GetClassName() => TypeDescriptor.GetClassName(this, true);

    TypeConverter ICustomTypeDescriptor.GetConverter() => TypeDescriptor.GetConverter(this, true);

    EventDescriptor? ICustomTypeDescriptor.GetDefaultEvent() => TypeDescriptor.GetDefaultEvent(this, true);

    PropertyDescriptor? ICustomTypeDescriptor.GetDefaultProperty() => TypeDescriptor.GetDefaultProperty(this, true);

    object? ICustomTypeDescriptor.GetEditor(Type editorBaseType) => TypeDescriptor.GetEditor(this, editorBaseType, true);

    EventDescriptorCollection ICustomTypeDescriptor.GetEvents(Attribute[]? attributes) =>
        TypeDescriptor.GetEvents(this, attributes, true);

    EventDescriptorCollection ICustomTypeDescriptor.GetEvents() => TypeDescriptor.GetEvents(this, true);

    string? ICustomTypeDescriptor.GetComponentName() => TypeDescriptor.GetComponentName(this, true);
}
