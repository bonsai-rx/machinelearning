using Bonsai.Expressions;
using System.Reactive.Linq;
using System.Threading.Tasks;
using System.Xml;

namespace Bonsai.ML.LinearDynamicalSystems.Tests;

internal static class WorkflowHelper
{
    public static async Task RunWorkflow(string path, params (string name, object value)[] properties)
    {
        using var reader = XmlReader.Create(path);
        var workflowBuilder = (WorkflowBuilder)WorkflowBuilder.Serializer.Deserialize(reader)!;
        for (int i = 0; i < properties.Length; i++)
        {
            workflowBuilder.Workflow.SetWorkflowProperty(properties[i].name, properties[i].value);
        }
        await workflowBuilder.Workflow.BuildObservable();
    }
}
