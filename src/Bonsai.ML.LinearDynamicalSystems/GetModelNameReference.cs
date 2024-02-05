using System.ComponentModel;
using System;
using System.Reactive.Linq;

namespace Bonsai.ML.LinearDynamicalSystems
{
    /// <summary>
    /// Gets the name of a Bonsai.ML model for reference
    /// </summary>
    [Description("Gets the name of a Bonsai.ML model")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class GetModelNameReference : INamedElement
    {

        /// <summary>
        /// The name of the model
        /// </summary>    
        [TypeConverter(typeof(ModelNameReferenceConverter))]
        [Description("Name of the model")]
        public string Name { get; set; }

        public IObservable<string> Process()
        {
    		return Observable.Defer(() => Observable.Return(
    			Name
            ));
        }
    }
}
