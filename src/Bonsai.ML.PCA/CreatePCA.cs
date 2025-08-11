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

        public double InitialVariance { get; set; } = 1.0;
        public int Iterations { get; set; } = 100;
        public double Tolerance { get; set; } = 1e-5;

        [XmlIgnore]
        public Generator? Generator { get; set; } = null;

        internal IEnumerable<string> GetModelProperties()
        {
            yield return nameof(NumComponents);
            yield return nameof(ModelType);

            if (ModelType == PCAModelType.ProbabilisticPCA)
            {
                yield return nameof(InitialVariance);
                yield return nameof(Iterations);
                yield return nameof(Tolerance);
                yield return nameof(Generator);
            }
            }
        }

        private static PCABaseModel CreateModel(CreatePCA instance)
        {
            return instance.ModelType switch
            {
                PCAModelType.PCA => new PCA(instance.NumComponents),
                PCAModelType.ProbabilisticPCA => new PPCA(
                    instance.NumComponents,
                    instance.InitialVariance,
                    instance.Generator,
                    instance.Iterations,
                    instance.Tolerance),
                _ => throw new NotSupportedException($"Model type {instance.ModelType} is not supported."),
            };
        }

        private static Type GetModelType(PCAModelType modelType)
        {
            return modelType switch
            {
                PCAModelType.PCA => typeof(PCA),
                PCAModelType.ProbabilisticPCA => typeof(PPCA),
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
