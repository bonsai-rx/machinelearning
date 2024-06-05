using System.ComponentModel;
using System;
using System.Reactive.Linq;

namespace Bonsai.ML
{
	/// <summary>
	/// Represents an operator that creates a reference for a named model.
	/// </summary>
	[Combinator]
	[Description("Creates a reference for a named model.")]
	[WorkflowElementCategory(ElementCategory.Source)]
	public class CreateModelReference : INamedElement
	{
        /// <summary>
        /// Gets or sets the name of the model to reference.
        /// </summary>
        [Description("The name of the model to reference.")]
        public string Name { get ; set; }

        /// <summary>
        /// Generates an observable sequence that contains the model reference object.
        /// </summary>
        /// <returns>
        /// A sequence containing a single instance of the <see cref="ModelReference"/>
        /// class.
        /// </returns>
        public IObservable<ModelReference> Process()
        {
            return Observable.Defer(() => Observable.Return(new ModelReference(Name)));
        }
	}
}
