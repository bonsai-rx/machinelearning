using System;
using System.ComponentModel;

namespace Bonsai.ML.Torch.NeuralNets;

internal class ModuleCombinatorPropertyDescriptor(PropertyDescriptor descriptor) : PropertyDescriptor(descriptor)
{
    private readonly PropertyDescriptor descriptor = descriptor;

    public override Type ComponentType => descriptor.ComponentType;

    public override bool IsReadOnly => false;

    public override Type PropertyType => typeof(Type);

    public override bool CanResetValue(object component)
    {
        return true;
    }

    public override object GetValue(object component)
    {
        component = descriptor.GetValue(component);
        return component?.GetType();
    }

    public override void ResetValue(object component)
    {
        descriptor.SetValue(component, null);
    }

    public override void SetValue(object component, object value)
    {
        var currentValue = descriptor.GetValue(component);
        var newValue = Activator.CreateInstance((Type)value);

        var newProperties = TypeDescriptor.GetProperties(newValue);
        var currentProperties = TypeDescriptor.GetProperties(currentValue);
        foreach (PropertyDescriptor property in newProperties)
        {
            var mergeProperty = currentProperties[property.Name];
            if (mergeProperty?.PropertyType == property.PropertyType)
            {
                var propertyValue = mergeProperty.GetValue(currentValue);
                property.SetValue(newValue, propertyValue);
            }
        }

        descriptor.SetValue(component, newValue);
    }

    public override bool ShouldSerializeValue(object component)
    {
        return true;
    }
}