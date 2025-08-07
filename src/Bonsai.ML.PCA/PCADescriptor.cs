using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Collections.Generic;
using System.Linq;
using Bonsai;
using static TorchSharp.torch;

namespace Bonsai.ML.PCA
{
    public class PCADescriptor : CustomTypeDescriptor
    {
        private readonly object _instance;

        public PCADescriptor(ICustomTypeDescriptor parent, object instance) : base(parent)
        {
            _instance = instance;
        }

        public override PropertyDescriptorCollection GetProperties(Attribute[] attributes)
        {
            var allProperties = base.GetProperties(attributes);

            if (_instance is CreatePCA createPCA)
            {
                var modelProperties = new HashSet<string>(createPCA.GetModelProperties());
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
}
