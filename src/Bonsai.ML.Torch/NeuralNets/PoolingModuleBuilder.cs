using System;
using System.Linq;
using System.Linq.Expressions;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Xml.Serialization;
using Bonsai.Expressions;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System.Reflection;

namespace Bonsai.ML.Torch.NeuralNets;

/// <summary>
/// Represents an operator that creates a torch module for pooling operations.
/// </summary>
[XmlInclude(typeof(Pooling.AdaptiveAvgPool1d))]
[XmlInclude(typeof(Pooling.AdaptiveAvgPool2d))]
[XmlInclude(typeof(Pooling.AdaptiveAvgPool3d))]
[XmlInclude(typeof(Pooling.AdaptiveMaxPool1d))]
[XmlInclude(typeof(Pooling.AdaptiveMaxPool2d))]
[XmlInclude(typeof(Pooling.AdaptiveMaxPool3d))]
[DefaultProperty(nameof(PoolingModule))]
[Combinator]
[Description("Creates a torch module for pooling operations.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class PoolingModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    /// <summary>
    /// Initializes a new instance of the <see cref="PoolingModuleBuilder"/> class.
    /// </summary>
    public PoolingModuleBuilder()
    {
        Module = new Pooling.AdaptiveAvgPool1d();
    }
    
    /// <summary>
    /// Gets or sets the specific pooling module to create.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Module")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The specific pooling module to create.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object PoolingModule
    {
        get => Module;
        set => Module = value;
    }
}
