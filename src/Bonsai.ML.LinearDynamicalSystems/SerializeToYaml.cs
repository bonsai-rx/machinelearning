namespace Bonsai.ML.LinearDynamicalSystems
{
    /// <summary>
    /// Serializes a sequence of data model objects into YAML strings.
    /// </summary>
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Transform)]
    [System.ComponentModel.Description("Serializes a sequence of data model objects into YAML strings.")]
    public class SerializeToYaml
    {
        private System.IObservable<string> Process<T>(System.IObservable<T> source)
        {
            return System.Reactive.Linq.Observable.Defer(() =>
            {
                var serializer = new YamlDotNet.Serialization.SerializerBuilder().Build();
                return System.Reactive.Linq.Observable.Select(source, value => serializer.Serialize(value)); 
            });
        }
    }
}
