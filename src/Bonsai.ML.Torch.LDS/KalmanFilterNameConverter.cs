using Bonsai.Expressions;
using System.Linq;
using System.ComponentModel;

namespace Bonsai.ML.Torch.LDS;

/// <summary>
/// Provides a type converter to select the name of an existing Kalman filter model in the workflow.
/// </summary>
public class KalmanFilterNameConverter : StringConverter
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
                              where managedModelNode != null && managedModelNode is CreateKalmanFilter
                              let createKalmanFilter = (CreateKalmanFilter)managedModelNode
                              where createKalmanFilter != null && !string.IsNullOrEmpty(createKalmanFilter.Name)
                              select createKalmanFilter.Name)
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