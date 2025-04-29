using Bonsai;
using Bonsai.Expressions;
using System.Linq;
using System.ComponentModel;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Provides a type converter to display a list of available point process models.
/// </summary>
public class PointProcessModelNameConverter : StringConverter
{
    /// <inheritdoc/>
    public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
    {
        return true;
    }

    /// <inheritdoc/>
    public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
    {
        if (context != null)
        {
            var workflowBuilder = (WorkflowBuilder)context.GetService(typeof(WorkflowBuilder));
            if (workflowBuilder != null)
            {
                var models = (from builder in workflowBuilder.Workflow.Descendants()
                                    where builder.GetType() != typeof(DisableBuilder)
                                    let managedModelNode = ExpressionBuilder.GetWorkflowElement(builder)
                                    where managedModelNode != null && (managedModelNode is CreatePointProcessModel || managedModelNode is LoadPointProcessModel) 
                                    let createPointProcessModel = (IManagedPointProcessModelNode)managedModelNode
                                    where createPointProcessModel != null && !string.IsNullOrEmpty(createPointProcessModel.Name)
                                    select createPointProcessModel.Name)
                                    .Distinct()
                                    .ToList();
                if (models.Count > 0)
                {
                    return new StandardValuesCollection(models);
                }
            }
        }

        return new StandardValuesCollection(new string[] { });
    }
}