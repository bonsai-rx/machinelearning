using System.ComponentModel;
using System;
using System.Reactive.Linq;

namespace Bonsai.ML.LinearDynamicalSystems
{
    /// <summary>
    /// Provides a Bonsai.ML model with a name that can be referenced
    /// </summary>
    [Description("Name of a Bonsai.ML model")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class CreateModelNameReference : INamedElement
    {

        /// <summary>
        /// The name of the model
        /// </summary>
        [Description("Name of the model")]
        public string Name { get ; set; }

        public IObservable<string> Process()
        {
    		return Observable.Defer(() => Observable.Return(
    			Name
            ));
        }
    }
}
