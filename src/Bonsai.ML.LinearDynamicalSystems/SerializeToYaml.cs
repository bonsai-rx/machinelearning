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

        public System.IObservable<string> Process(System.IObservable<Model> source)
        {
            return Process<Model>(source);
        }

        public System.IObservable<string> Process(System.IObservable<Observation2D> source)
        {
            return Process<Observation2D>(source);
        }

        public System.IObservable<string> Process(System.IObservable<State> source)
        {
            return Process<State>(source);
        }

        public System.IObservable<string> Process(System.IObservable<StateEstimate> source)
        {
            return Process<StateEstimate>(source);
        }

        public System.IObservable<string> Process(System.IObservable<EstimateWithUncertainty> source)
        {
            return Process<EstimateWithUncertainty>(source);
        }

        public System.IObservable<string> Process(System.IObservable<Ellipse> source)
        {
            return Process<Ellipse>(source);
        }
    }
}
