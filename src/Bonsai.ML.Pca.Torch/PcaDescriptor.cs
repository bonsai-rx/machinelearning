using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.Pca.Torch;

public class PcaDescriptor : CustomTypeDescriptor
{
    private readonly object _instance;

    public PcaDescriptor(ICustomTypeDescriptor parent, object instance) : base(parent)
    {
        _instance = instance;
    }

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

    public override PropertyDescriptorCollection GetProperties()
        => GetProperties(null);
}
