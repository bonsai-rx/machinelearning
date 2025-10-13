using Bonsai.Expressions;
using System.Reactive.Linq;
using System.Xml;

namespace Bonsai.ML.Tests.Utilities;

/// <summary>
/// Helper class to run Bonsai workflows in unit tests.
/// </summary>
public static class WorkflowHelper
{
    /// <summary>
    /// Runs a Bonsai workflow from the specified file path and sets the specified workflow properties.
    /// </summary>
    /// <param name="path"></param>
    /// <param name="properties"></param>
    /// <returns></returns>
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
