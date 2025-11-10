using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Provides a custom type descriptor for PCA model creation.
/// </summary>
/// <remarks>
/// Initializes a new instance of the <see cref="PcaDescriptor"/> class.
/// </remarks>
/// <param name="parent"></param>
/// <param name="instance"></param>
public class PcaDescriptor(ICustomTypeDescriptor parent, object instance) : CustomTypeDescriptor(parent)
{
    private readonly object _instance = instance;

    /// <inheritdoc/>
    public override PropertyDescriptorCollection GetProperties(Attribute[] attributes)
    {
        var allProperties = base.GetProperties(attributes);

        if (_instance is CreatePca createPca)
        {
            var modelProperties = new HashSet<string>(createPca.GetModelProperties());
            var filtered = allProperties.Cast<PropertyDescriptor>()
                .Where(p => modelProperties.Contains(p.Name))
                .ToArray();
            return new PropertyDescriptorCollection(filtered);
        }

        return allProperties;
    }

    /// <inheritdoc/>
    public override PropertyDescriptorCollection GetProperties()
        => GetProperties([]);
}
