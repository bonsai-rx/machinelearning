using System.Reactive.Linq;
using System.Reactive;
using System;

namespace Bonsai.ML.LinearDynamicalSystems
{
    [System.ComponentModel.DefaultProperty("Name")]
    // [Combinator()]
    // [WorkflowElementCategory(ElementCategory.Source)]
    public class AddModel : Source<Unit>
    {
        private string _name = null;
        // private string _prevName = null;

        /// <summary>
        /// Sets the model name to be referenced.
        /// </summary>
        public string Name { 
            get => _name; 
            set {
                // _prevName = _name;
                ModelCollection.RemoveModelName(_name);
                _name = value?.Replace(" ", "");
                ModelCollection.AddModelName(_name);
                // ModelCollection.RemoveModelName(_prevName);
            }
        }

        public override IObservable<Unit> Generate()
        {
            return Observable.Return(Unit.Default);
        }

        public IObservable<Unit> Generate<T>(IObservable<T> source)
        {
            return Observable.Select(source, value => Unit.Default);
        }

        // void IDisposable.Dispose() {
        //     ModelCollection.RemoveModelName(_name);
        // }
    }
}
