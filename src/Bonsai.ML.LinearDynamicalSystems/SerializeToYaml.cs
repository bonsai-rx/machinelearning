using System;
using System.ComponentModel;
using System.Reactive.Linq;
using YamlDotNet.Serialization;

namespace Bonsai.ML.LinearDynamicalSystems
{
    /// <summary>
    /// Serializes a sequence of data model objects into YAML strings.
    /// </summary>
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Transform)]
    [Description("Serializes a sequence of data model objects into YAML strings.")]
    public class SerializeToYaml
    {
        private IObservable<string> Process<T>(IObservable<T> source)
        {
            return Observable.Defer(() =>
            {
                var serializer = new SerializerBuilder().Build();
                return Observable.Select(source, value => serializer.Serialize(value)); 
            });
        }

        public IObservable<string> Process(IObservable<KFKModelParameters> source)
        {
            return Process<KFKModelParameters>(source);
        }

        public IObservable<string> Process(IObservable<Observation2D> source)
        {
            return Process<Observation2D>(source);
        }

        public IObservable<string> Process(IObservable<State> source)
        {
            return Process<State>(source);
        }

        public IObservable<string> Process(IObservable<Kinematics> source)
        {
            return Process<Kinematics>(source);
        }

        public IObservable<string> Process(IObservable<KinematicComponent> source)
        {
            return Process<KinematicComponent>(source);
        }
    }
}
