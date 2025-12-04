using Bonsai.Expressions;
using System;
using System.ComponentModel;
using System.Linq;
using System.Linq.Expressions;
using System.Collections.Generic;
using System.Reflection;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents a base class for module combinator builders.
/// </summary>
public abstract class ModuleCombinatorBuilder : ExpressionBuilder, ICustomTypeDescriptor, INamedElement
{
    /// <inheritdoc/>
    string INamedElement.Name => $"Module.{GetElementDisplayName(Module)}";

    internal object Module { get; set; }

    /// <inheritdoc/>
    public override Expression Build(IEnumerable<Expression> arguments)
    {
        // We want to return an expression that constructs the module
        var module = Module.GetType();

        // arguments can either be empty or contain a single argument
        if (!arguments.Any())
        {
            // if empty, we call the non generic Process method
            var methodInfo = module.GetMethods(BindingFlags.Public | BindingFlags.Instance).First(m => m.Name == "Process" && !m.IsGenericMethod);
            return Expression.Call(
                Expression.Constant(Module, module),
                methodInfo
            );
        }
        else
        {
            // if there is an argument, we call the generic Process method
            var argument = arguments.First();
            var argumentType = argument.Type.GetGenericArguments()[0];
            var methodInfo = module.GetMethods(BindingFlags.Public | BindingFlags.Instance).First(m => m.Name == "Process" && m.IsGenericMethodDefinition && m.GetGenericArguments().Length == 1);
            var genericMethodInfo = methodInfo.MakeGenericMethod(argumentType);
            return Expression.Call(
                Expression.Constant(Module, module),
                genericMethodInfo,
                argument
            );
        }
    }

    AttributeCollection ICustomTypeDescriptor.GetAttributes()
    {
        var attributes = TypeDescriptor.GetAttributes(GetType());
        var defaultProperty = TypeDescriptor.GetDefaultProperty(GetType());
        if (defaultProperty != null)
        {
            var instance = defaultProperty.GetValue(this);
            var instanceAttributes = TypeDescriptor.GetAttributes(instance);
            if (instanceAttributes[typeof(DescriptionAttribute)] is DescriptionAttribute description)
            {
                return AttributeCollection.FromExisting(attributes, description);
            }
        }

        return attributes;
    }

    string ICustomTypeDescriptor.GetClassName()
    {
        return TypeDescriptor.GetClassName(GetType());
    }

    string ICustomTypeDescriptor.GetComponentName()
    {
        return null;
    }

    TypeConverter ICustomTypeDescriptor.GetConverter()
    {
        return TypeDescriptor.GetConverter(GetType());
    }

    EventDescriptor ICustomTypeDescriptor.GetDefaultEvent()
    {
        return null;
    }

    PropertyDescriptor ICustomTypeDescriptor.GetDefaultProperty()
    {
        var defaultProperty = TypeDescriptor.GetDefaultProperty(GetType());
        return defaultProperty != null ? new ModuleCombinatorPropertyDescriptor(defaultProperty) : null;
    }

    object ICustomTypeDescriptor.GetEditor(Type editorBaseType)
    {
        return TypeDescriptor.GetEditor(GetType(), editorBaseType);
    }

    EventDescriptorCollection ICustomTypeDescriptor.GetEvents()
    {
        return EventDescriptorCollection.Empty;
    }

    EventDescriptorCollection ICustomTypeDescriptor.GetEvents(Attribute[] attributes)
    {
        return EventDescriptorCollection.Empty;
    }

    PropertyDescriptorCollection ICustomTypeDescriptor.GetProperties()
    {
        return ((ICustomTypeDescriptor)this).GetProperties([]);
    }

    PropertyDescriptorCollection ICustomTypeDescriptor.GetProperties(Attribute[] attributes)
    {
        var baseProperties = TypeDescriptor.GetProperties(GetType(), attributes);
        var defaultProperty = TypeDescriptor.GetDefaultProperty(GetType());
        if (defaultProperty != null)
        {
            var instance = defaultProperty.GetValue(this);
            var instanceProperties = TypeDescriptor.GetProperties(instance, attributes);
            var properties = new PropertyDescriptor[baseProperties.Count + instanceProperties.Count];
            for (int i = 0; i < baseProperties.Count; i++)
            {
                var baseProperty = baseProperties[i];
                if (baseProperty == defaultProperty)
                {
                    baseProperty = new ModuleCombinatorPropertyDescriptor(defaultProperty);
                }

                properties[i] = baseProperty;
            }

            for (int i = 0; i < instanceProperties.Count; i++)
            {
                var expandedProperty = instanceProperties[i];
                properties[i + baseProperties.Count] = expandedProperty;
            }
            return new PropertyDescriptorCollection(properties);
        }

        return baseProperties;
    }

    object ICustomTypeDescriptor.GetPropertyOwner(PropertyDescriptor pd)
    {
        return pd?.ComponentType.IsAssignableFrom(GetType()) == true ? this : Module;
    }
}