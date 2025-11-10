using System;
using System.ComponentModel;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Provides a custom type description provider for PCA models.
/// </summary>
/// <remarks>
/// Initializes a new instance of the <see cref="PcaDescriptionProvider"/> class.
/// </remarks>
/// <param name="baseProvider"></param>
public class PcaDescriptionProvider(TypeDescriptionProvider baseProvider) : TypeDescriptionProvider(baseProvider)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="PcaDescriptionProvider"/> class.
    /// </summary>
    public PcaDescriptionProvider() : this(TypeDescriptor.GetProvider(typeof(object))) { }

    /// <inheritdoc/>
    public override ICustomTypeDescriptor GetTypeDescriptor(Type objectType, object instance)
    {
        var defaultDescriptor = base.GetTypeDescriptor(objectType, instance);
        return new PcaDescriptor(defaultDescriptor, instance);
    }
}