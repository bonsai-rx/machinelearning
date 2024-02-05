using Bonsai.Expressions;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace Bonsai.ML.LinearDynamicalSystems
{
    public class ModelNameReferenceConverter : StringConverter
    {
        public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
        {
            return true;
        }

        public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
        {         
            if (context != null)
            {
                var workflowBuilder = (WorkflowBuilder)context.GetService(typeof(WorkflowBuilder));
                if (workflowBuilder != null)
                {
                    var modelNames = (from builder in workflowBuilder.Workflow.Descendants()
                                     where builder is not DisableBuilder
                                     let createModel = ExpressionBuilder.GetWorkflowElement(builder) as CreateModelNameReference
                                     where createModel != null && !string.IsNullOrEmpty(createModel.Name)
                                     select createModel.Name)
                                     .Distinct()
                                     .ToList();
                    if (modelNames.Count > 0)
                    {
                        return new StandardValuesCollection(modelNames);
                    }
                }
            }

            return new StandardValuesCollection(new List<string>());
        }
    }
}