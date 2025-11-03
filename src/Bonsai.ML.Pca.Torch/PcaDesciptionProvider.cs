using System;
using System.ComponentModel;

namespace Bonsai.ML.Pca.Torch;

class PcaDescriptionProvider : TypeDescriptionProvider
{
    private readonly TypeDescriptionProvider _baseProvider;

    public PcaDescriptionProvider() : this(TypeDescriptor.GetProvider(typeof(object))) { }

    public PcaDescriptionProvider(TypeDescriptionProvider baseProvider)
        : base(baseProvider)
    {
        _baseProvider = baseProvider;
    }

    public override ICustomTypeDescriptor GetTypeDescriptor(Type objectType, object instance)
    {
        var defaultDescriptor = base.GetTypeDescriptor(objectType, instance);
        return new PcaDescriptor(defaultDescriptor, instance);
    }
}