using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Linq.Expressions;
using Bonsai.Expressions;
using System.Linq;
using System.Reflection;

namespace Bonsai.ML.PCA
{
    [Combinator]
    [WorkflowElementCategory(ElementCategory.Source)]
    [TypeDescriptionProvider(typeof(PCADescriptionProvider))]
    public class CreatePCA : ZeroArgumentExpressionBuilder
    {
        public int NumComponents { get; set; } = 2;

        [RefreshProperties(RefreshProperties.All)]
        public PCAModelType ModelType { get; set; } = PCAModelType.PCA;
        public double Variance { get; set; } = 1.0;

        internal IEnumerable<string> GetModelProperties()
        {
            yield return nameof(NumComponents);
            yield return nameof(ModelType);

            if (ModelType == PCAModelType.ProbabilisticPCA)
            {
                yield return "Variance";
            }
        }

        private static PCABaseModel CreateModel(CreatePCA instance)
        {
            return instance.ModelType switch
            {
                PCAModelType.PCA => new PCA(instance.NumComponents),
                _ => throw new NotSupportedException($"Model type {instance.ModelType} is not supported."),
            };
        }

        private static Type GetModelType(PCAModelType modelType)
        {
            return modelType switch
            {
                PCAModelType.PCA => typeof(PCA),
                _ => throw new NotSupportedException($"Model type {modelType} is not supported."),
            };
        }

        public override Expression Build(IEnumerable<Expression> arguments)
        {
            var processMethod = GetType().GetMethod(
                nameof(Process),
                BindingFlags.NonPublic | BindingFlags.Static);
            processMethod = processMethod.MakeGenericMethod(GetModelType(ModelType));
            return Expression.Call(processMethod, Expression.Constant(this));
        }

        private static IObservable<T> Process<T>(CreatePCA instance) where T : PCABaseModel
        {
            return Observable.Return((T)CreateModel(instance));
        }
    }
}
