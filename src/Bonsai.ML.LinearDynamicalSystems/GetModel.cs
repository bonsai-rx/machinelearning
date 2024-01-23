using System.ComponentModel;
using System.Reactive.Linq;
using System.Xml.Serialization;

namespace Bonsai.ML.LinearDynamicalSystems
{
    [DefaultProperty("Name")]
    // [Source]
    [WorkflowElementCategory(ElementCategory.Source)]
    // [XmlType(Namespace = Constants.XmlNamespace)]
    public class GetModel : Source<string>
    {
        /// <summary>
        /// Gets the name of the model to be referenced.
        /// </summary>
        [TypeConverter(typeof(ModelCollection))]
        [Description("The name of the model to be referenced.")]
        public string Name
        { 
            get => _name; 
            set => _name = value;
        }

        private string _name;
        
        public override System.IObservable<string> Generate()
        {
            return Observable.Return(_name);
        }
        
        public System.IObservable<string> Generate<T>(System.IObservable<T> source)
        {
            return Observable.Select(source, value => _name);
        }
    }
}