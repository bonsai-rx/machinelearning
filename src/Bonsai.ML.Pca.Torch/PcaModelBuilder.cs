using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Xml.Serialization;
using Bonsai.Expressions;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Represents an abstract PCA model builder.
/// </summary>
public abstract class PcaModelBuilder<T>(IPcaModelProvider operatorInstance) : SingleArgumentExpressionBuilder, ICustomTypeDescriptor
{
    private readonly IPcaModelProvider _operator = operatorInstance;

    /// <summary>
    /// The PCA model.
    /// </summary>
    [XmlIgnore]
    [Description("The PCA model.")]
    public IPcaBaseModel? Model
    {
        get => _operator.Model;
        set => _operator.Model = value;
    }

    /// <summary>
    /// Determines whether the model is available in the input expression (and thus Model property should not be exposed) or if it should be explicitly set as a property of the operator.
    /// </summary>
    [Browsable(false)]
    [XmlIgnore]
    [RefreshProperties(RefreshProperties.All)]
    public bool HasModel { get; private set; } = true;

    /// <inheritdoc/>
    public override Expression Build(IEnumerable<Expression> arguments)
    {
        var input = arguments?.FirstOrDefault();
        MethodInfo processMethod;
        if (input is null)
        {
            HasModel = true;
            processMethod = typeof(T).GetMethod("Process", [typeof(IObservable<Tensor>)]);
            return Expression.Call(Expression.Constant(_operator), processMethod!, [input]);
        }

        var obsType = input.Type;
        var t = obsType.GetGenericArguments().Single();

        HasModel = !(t.IsGenericType && t.FullName?.StartsWith("System.Tuple`") == true);

        if (!HasModel)
        {
            var args = t.GetGenericArguments();
            if (args.Length != 2 || (!typeof(IPcaBaseModel).IsAssignableFrom(args[0]) && !typeof(IPcaBaseModel).IsAssignableFrom(args[1])))
                throw new InvalidOperationException("The input type is not valid. Expected an observable sequence of tuples containing a PCA model and a tensor.");
        }

        processMethod = HasModel
            ? typeof(T).GetMethod("Process", [typeof(IObservable<Tensor>)])
            : typeof(T).GetMethod("Process", [typeof(IObservable<>).MakeGenericType(t)]);

        return Expression.Call(Expression.Constant(_operator), processMethod!, [input]);
    }


    PropertyDescriptorCollection ICustomTypeDescriptor.GetProperties(Attribute[]? attributes)
    {
        var props = TypeDescriptor.GetProperties(this, attributes, true);
        if (HasModel) return props;

        var filtered = props.Cast<PropertyDescriptor>()
                            .Where(p => p.Name != nameof(Model))
                            .ToArray();

        return new PropertyDescriptorCollection(filtered);
    }

    AttributeCollection ICustomTypeDescriptor.GetAttributes() => TypeDescriptor.GetAttributes(this, true);
    string? ICustomTypeDescriptor.GetClassName() => TypeDescriptor.GetClassName(this, true);
    string? ICustomTypeDescriptor.GetComponentName() => TypeDescriptor.GetComponentName(this, true);
    TypeConverter ICustomTypeDescriptor.GetConverter() => TypeDescriptor.GetConverter(this, true);
    EventDescriptor? ICustomTypeDescriptor.GetDefaultEvent() => TypeDescriptor.GetDefaultEvent(this, true);
    PropertyDescriptor? ICustomTypeDescriptor.GetDefaultProperty() => TypeDescriptor.GetDefaultProperty(this, true);
    object? ICustomTypeDescriptor.GetEditor(Type editorBaseType) => TypeDescriptor.GetEditor(this, editorBaseType, true);
    EventDescriptorCollection ICustomTypeDescriptor.GetEvents(Attribute[]? attributes) => TypeDescriptor.GetEvents(this, attributes, true);
    EventDescriptorCollection ICustomTypeDescriptor.GetEvents() => TypeDescriptor.GetEvents(this, true);
    PropertyDescriptorCollection ICustomTypeDescriptor.GetProperties() => ((ICustomTypeDescriptor)this).GetProperties(Array.Empty<Attribute>());
    object ICustomTypeDescriptor.GetPropertyOwner(PropertyDescriptor? pd) => this;
}
