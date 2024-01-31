using System;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using Bonsai.Expressions;
using System.Linq.Expressions;
using Bonsai;
using System.ComponentModel;
using System.Reactive;

namespace Bonsai.ML.LinearDynamicalSystems
{
    // [WorkflowElementIcon(nameof(ModelSubject))]
    // // [Source()]
    // [WorkflowElementCategory(ElementCategory.Source)]
    // public class ModelSubject : SubjectBuilder
    // {

    //     private string _name = null;

    //     [Category("Subject")]
    //     public string Name { 
    //         get => _name; 
    //         set => _name = value;
    //     }

    //     /// <inheritdoc/>
    //     protected override Expression BuildSubject(Expression expression)
    //     {
    //         Console.WriteLine("here build");
    //         var builderExpression = Expression.Constant(this);
    //         // var parameterType = expression.Type.GetGenericArguments()[0];
    //         return Expression.Call(builderExpression, nameof(CreateSubject), new[] { typeof(string) });
    //     }

    //     // public SubjectBuilder<string>CreateSubject();
    //     // public override IObservable<BehaviorSubject<string>> Generate()
    //     // {
    //     //     return Observable.Return(CreateSubject(Name));
    //     // }

    //     protected ISubject<string> CreateSubject()
    //     // ModelSubject<string> CreateSubject()
    //     {
    //         Console.WriteLine("here");
    //         // return new ModelSubject<string>(_name);
    //         return new BehaviorSubject<string>(_name);
    //         // return new ModelSubject<string>.Subject();
    //     }
        

    // }

    // public class ModelSubject<T> : SubjectBuilder<string>
    // {
    //     /// <summary>
    //     /// Creates a shared subject that broadcasts the latest value from other observable
    //     /// sequences to all subscribed and future observers.
    //     /// </summary>
    //     /// <returns>A new instance of <see cref="ISubject{T}"/>.</returns>
    //     protected override ISubject<string> CreateSubject()
    //     {
    //         return new Subject();
    //     }

    //     internal class Subject : ISubject<string>, IDisposable
    //     {
    //         readonly ReplaySubject<string> subject = new ReplaySubject<string>(1);

    //         public void OnCompleted()
    //         {
    //         }

    //         public void OnError(Exception error)
    //         {
    //         }

    //         public void OnNext(string value)
    //         {
    //             subject.OnNext(value);
    //         }

    //         public void Dispose()
    //         {
    //             subject.Dispose();
    //         }
    //     }
    // }

    /// <summary>
    /// Represents an expression builder that broadcasts the latest value of an observable
    /// sequence to all subscribed and future observers using a shared subject.
    /// </summary>
    [WorkflowElementIcon(nameof(ModelSubject))]
    // [XmlType(Namespace = Constants.ReactiveXmlNamespace)]
    // [Combinator()]
    [WorkflowElementCategory(ElementCategory.Source)]
    [Description("Broadcasts the latest value of an observable sequence to all subscribed and future observers using a shared subject.")]
    public class ModelSubject : Source<Unit>
    {

        private BehaviorSubject<string> _subject;

        private string _name = null;

        [Category("Subject")]
        public string Name { 
            get => _name; 
            set {
                _name = value;
                _subject.Dispose();
                _subject = new BehaviorSubject<string>(_name);
            }
        }

        public ModelSubject()
        {
            _subject = new BehaviorSubject<string>(_name);
        }

        public override IObservable<Unit> Generate()
        {
            return Observable.Return(new Unit());
        }

        /// <inheritdoc/>
        // protected Expression BuildSubject(Expression expression)
        // {
        //     Console.WriteLine("buildsubject");
        //     var builderExpression = Expression.Constant(this);
        //     // var parameterType = expression.Type.GetGenericArguments()[0];
        //     var parameterType = typeof(string);
        //     return Expression.Call(builderExpression, nameof(CreateSubject), new[] { parameterType });
        // }

        // protected override ISubject<string> CreateSubject()
        // {
        //     Console.WriteLine("createsubject");
        //     return new Subject<string>();
        // }
    }
}
