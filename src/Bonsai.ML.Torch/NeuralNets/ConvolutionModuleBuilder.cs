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
/// Creates a convolution module.
/// </summary>
[XmlInclude(typeof(Convolution.Convolution))]
[XmlInclude(typeof(Convolution.Fold))]
[XmlInclude(typeof(Convolution.Unfold))]
[DefaultProperty(nameof(ConvolutionModule))]
[Combinator]
[Description("Creates a convolution module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class ConvolutionModuleBuilder : ModuleCombinatorBuilder, INamedElement
{
    /// <inheritdoc/>
    public override Range<int> ArgumentRange => Range.Create(0, 1);

    /// <summary>
    /// Initializes a new instance of the <see cref="ConvolutionModuleBuilder"/> class.
    /// </summary>
    public ConvolutionModuleBuilder()
    {
        Module = new Convolution.Fold();
    }
    
    /// <summary>
    /// Gets or sets the event parser used to filter and select event messages
    /// reported by the device.
    /// </summary>
    [DesignOnly(true)]
    [DisplayName("Type")]
    [Externalizable(false)]
    [RefreshProperties(RefreshProperties.All)]
    [Category(nameof(CategoryAttribute.Design))]
    [Description("The type of the device event message to select.")]
    [TypeConverter(typeof(ModuleTypeConverter))]
    public object ConvolutionModule
    {
        get => Module;
        set => Module = value;
    }

    string INamedElement.Name => $"Convolution.{GetElementDisplayName(ConvolutionModule)}";

    /// <inheritdoc/>
    public override Expression Build(IEnumerable<Expression> arguments)
    {
        // We want to return an expression that constructs the module
        // For now, the only module supported is the Fold module
        // We want to return the Process method of the Fold class
        var convolutionModule = ConvolutionModule.GetType();

        // arguments can either be empty or contain a single argument
        if (!arguments.Any())
        {
            // if empty, we call the non genericProcess method
            var methodInfo = convolutionModule.GetMethods(BindingFlags.Public | BindingFlags.Instance).First(m => m.Name == "Process" && !m.IsGenericMethod);
            return Expression.Call(
                Expression.Constant(ConvolutionModule, convolutionModule),
                methodInfo
            );
        }
        else
        {
            // if there is an argument, we call the generic Process method
            var argument = arguments.First();
            var argumentType = argument.Type.GetGenericArguments()[0];
            var methodInfo = convolutionModule.GetMethods(BindingFlags.Public | BindingFlags.Instance).First(m => m.Name == "Process" && m.IsGenericMethodDefinition && m.GetGenericArguments().Length == 1);
            var genericMethodInfo = methodInfo.MakeGenericMethod(argumentType);
            return Expression.Call(
                Expression.Constant(ConvolutionModule, convolutionModule),
                genericMethodInfo,
                argument
            );
        }
    }
}
