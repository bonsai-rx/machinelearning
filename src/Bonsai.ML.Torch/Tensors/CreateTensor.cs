using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Linq.Expressions;
using System.Reactive.Linq;
using System.Reflection;
using System.Xml.Serialization;
using Bonsai.Expressions;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Tensors
{
    /// <summary>
    /// Creates a tensor from the specified values. Uses Python-like syntax to specify the tensor values. For example, a 2x2 tensor can be created with the following values: "[[1, 2], [3, 4]]".
    /// </summary>
    [Combinator]
    [Description("Creates a tensor from the specified values. Uses Python-like syntax to specify the tensor values. For example, a 2x2 tensor can be created with the following values: \"[[1, 2], [3, 4]]\".")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class CreateTensor : ExpressionBuilder
    {
        Range<int> argumentRange = new Range<int>(0, 1);

        /// <inheritdoc/>
        public override Range<int> ArgumentRange => argumentRange;

        /// <summary>
        /// The data type of the tensor elements.
        /// </summary>
        public TensorDataType Type
        {
            get => scalarType;
            set => scalarType = value;
        }

        private TensorDataType scalarType = TensorDataType.Float32;

        /// <summary>
        /// The values of the tensor elements. Uses Python-like syntax to specify the tensor values.
        /// </summary>
        public string Values
        {
            get => values;
            set
            {
                values = value.Replace("False", "false").Replace("True", "true");
            }
        }

        private string values = "[0]";

        /// <summary>
        /// The device on which to create the tensor.
        /// </summary>
        [XmlIgnore]
        public Device Device { get => device; set => device = value; }

        private Device device = null;

        private Expression BuildTensorFromArray(Array arrayValues, Type returnType)
        {
            var rank = arrayValues.Rank;
            var lengths = new int[rank];
            for (int i = 0; i < rank; i++)
            {
                lengths[i] = arrayValues.GetLength(i);
            }

            var arrayCreationExpression = Expression.NewArrayBounds(returnType, lengths.Select(len => Expression.Constant(len)).ToArray());
            var arrayVariable = Expression.Variable(arrayCreationExpression.Type, "array");
            var assignArray = Expression.Assign(arrayVariable, arrayCreationExpression);

            var assignments = new List<Expression>();
            for (int i = 0; i < values.Length; i++)
            {
                var indices = new Expression[rank];
                int temp = i;
                for (int j = rank - 1; j >= 0; j--)
                {
                    indices[j] = Expression.Constant(temp % lengths[j]);
                    temp /= lengths[j];
                }
                var value = Expression.Constant(arrayValues.GetValue(indices.Select(e => ((ConstantExpression)e).Value).Cast<int>().ToArray()));
                var arrayAccess = Expression.ArrayAccess(arrayVariable, indices);
                var assignArrayValue = Expression.Assign(arrayAccess, value);
                assignments.Add(assignArrayValue);
            }

            var tensorDataInitializationBlock = Expression.Block(
                arrayVariable,
                assignArray,
                Expression.Block(assignments),
                arrayVariable
            );

            var tensorCreationMethodInfo = typeof(TorchSharp.torch).GetMethod(
                "tensor", [
                    arrayVariable.Type,
                    typeof(ScalarType?),
                    typeof(Device),
                    typeof(bool),
                    typeof(string).MakeArrayType()
                ]
            );

            var tensorAssignment = Expression.Call(
                tensorCreationMethodInfo,
                tensorDataInitializationBlock,
                Expression.Constant(scalarType, typeof(ScalarType?)),
                Expression.Constant(device, typeof(Device)),
                Expression.Constant(false, typeof(bool)),
                Expression.Constant(null, typeof(string).MakeArrayType())
            );

            var tensorVariable = Expression.Variable(typeof(Tensor), "tensor");
            var assignTensor = Expression.Assign(tensorVariable, tensorAssignment);

            var buildTensor = Expression.Block(
                tensorVariable,
                assignTensor,
                tensorVariable
            );

            return buildTensor;
        }

        private Expression BuildTensorFromScalarValue(object scalarValue, Type returnType)
        {
            var valueVariable = Expression.Variable(returnType, "value");
            var assignValue = Expression.Assign(valueVariable, Expression.Constant(scalarValue, returnType));

            var tensorDataInitializationBlock = Expression.Block(
                valueVariable,
                assignValue,
                valueVariable
            );

            var tensorCreationMethodInfo = typeof(TorchSharp.torch).GetMethod(
                "tensor", [
                    valueVariable.Type,
                    typeof(Device),
                    typeof(bool)
                ]
            );

            var tensorCreationMethodArguments = new Expression[] {
                Expression.Constant(device, typeof(Device) ),
                Expression.Constant(false, typeof(bool) )
            };

            if (tensorCreationMethodInfo == null)
            {
                tensorCreationMethodInfo = typeof(TorchSharp.torch).GetMethod(
                    "tensor", [
                        valueVariable.Type,
                        typeof(ScalarType?),
                        typeof(Device),
                        typeof(bool)
                    ]
                );

                tensorCreationMethodArguments = tensorCreationMethodArguments.Prepend(
                    Expression.Constant(scalarType, typeof(ScalarType?))
                ).ToArray();
            }

            tensorCreationMethodArguments = tensorCreationMethodArguments.Prepend(
                    tensorDataInitializationBlock
                ).ToArray();

            var tensorAssignment = Expression.Call(
                tensorCreationMethodInfo,
                tensorCreationMethodArguments
            );

            var tensorVariable = Expression.Variable(typeof(Tensor), "tensor");
            var assignTensor = Expression.Assign(tensorVariable, tensorAssignment);

            var buildTensor = Expression.Block(
                tensorVariable,
                assignTensor,
                tensorVariable
            );

            return buildTensor;
        }

        /// <inheritdoc/>
        public override Expression Build(IEnumerable<Expression> arguments)
        {
            var returnType = Helpers.TensorDataTypeHelper.GetTypeFromTensorDataType(scalarType);
            var argTypes = arguments.Select(arg => arg.Type).ToArray();

            var methodInfoArgumentTypes = new Type[] {
                typeof(Tensor)
            };

            var methods = typeof(CreateTensor).GetMethods(BindingFlags.Public | BindingFlags.Instance)
                                    .Where(m => m.Name == "Process")
                                    .ToArray();

            var methodInfo = arguments.Count() > 0 ? methods.FirstOrDefault(m => m.IsGenericMethod)
                .MakeGenericMethod(
                    arguments
                        .First()
                        .Type
                        .GetGenericArguments()[0]
                ) : methods.FirstOrDefault(m => !m.IsGenericMethod);

            var tensorValues = Helpers.DataHelper.ParseString(values, returnType);
            var buildTensor = tensorValues is Array arrayValues ? BuildTensorFromArray(arrayValues, returnType) : BuildTensorFromScalarValue(tensorValues, returnType);
            var methodArguments = arguments.Count() == 0 ? [ buildTensor ] : arguments.Concat([ buildTensor ]);

            try
            {
                return Expression.Call(
                    Expression.Constant(this),
                    methodInfo,
                    methodArguments
                );
            }
            finally
            {
                values = Helpers.DataHelper.SerializeData(tensorValues).Replace("False", "false").Replace("True", "true");
                scalarType = Helpers.TensorDataTypeHelper.GetTensorDataTypeFromType(returnType);
            }
        }

        /// <summary>
        /// Returns an observable sequence that creates a tensor from the specified values.
        /// </summary>
        public IObservable<Tensor> Process(Tensor tensor)
        {
            return Observable.Return(tensor);
        }

        /// <summary>
        /// Returns an observable sequence that creates a tensor from the specified values for each element in the input sequence.
        /// </summary>
        public IObservable<Tensor> Process<T>(IObservable<T> source, Tensor tensor)
        {
            return Observable.Select(source, (_) => tensor);
        }
    }
}
