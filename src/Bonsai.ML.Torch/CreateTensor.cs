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
using Bonsai.ML.Data;
using TorchSharp;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Creates a tensor from the specified values.
    /// Uses Python-like syntax to specify the tensor values. 
    /// For example, a 2x2 tensor can be created with the following values: "[[1, 2], [3, 4]]".
    /// </summary>
    [Combinator]
    [Description("Creates a tensor from the specified values. Uses Python-like syntax to specify the tensor values. For example, a 2x2 tensor can be created with the following values: \"[[1, 2], [3, 4]]\".")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class CreateTensor : VariableArgumentExpressionBuilder
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CreateTensor"/> class.
        /// </summary>
        public CreateTensor()
            : base(minArguments: 0, maxArguments: 1)
        {
        }

        /// <summary>
        /// The data type of the tensor elements.
        /// </summary>
        [Description("The data type of the tensor elements.")]
        [TypeConverter(typeof(ScalarTypeConverter))]
        public ScalarType Type
        {
            get => _scalarType;
            set
            {
                if (ValidateUserInput(out var returnType, out var tensorData, out var validatedStringValues, scalarType: value))
                {
                    _returnType = returnType;
                    _tensorData = tensorData;
                    _stringValues = validatedStringValues;
                    _scalarType = value;
                }
                else
                {
                    throw new ArgumentException("Invalid input.");
                }
            }
        }
        private Type _returnType = typeof(float);
        private ScalarType _scalarType = ScalarType.Float32;

        /// <summary>
        /// The values of the tensor elements. 
        /// Uses Python-like syntax to specify the tensor values.
        /// For example: "[[1, 2], [3, 4]]".
        /// </summary>
        [Description("The values of the tensor elements. Uses Python-like syntax to specify the tensor values. For example: \"[[1, 2], [3, 4]]\".")]
        public string Values
        {
            get => _stringValues;
            set
            {
                if (ValidateUserInput(out var returnType, out var tensorData, out var validatedStringValues, stringValues: value))
                {
                    _returnType = returnType;
                    _tensorData = tensorData;
                    _stringValues = validatedStringValues;
                }
                else
                {
                    throw new ArgumentException("Invalid input.");
                }
            }
        }
        private object _tensorData = new float[] { 0 };
        private string _stringValues = "[0]";

        /// <summary>
        /// The device on which to create the tensor.
        /// </summary>
        [XmlIgnore]
        [Description("The device on which to create the tensor.")]
        public Device Device
        {
            get => _device;
            set => _device = value;
        }
        private Device _device = null;

        private bool ValidateUserInput(
            out Type returnType,
            out object tensorData,
            out string validatedStringValues,
            string stringValues = null, 
            ScalarType? scalarType = null
        )
        {
            stringValues ??= _stringValues;
            scalarType ??= _scalarType;

            try
            {
                returnType = ScalarTypeLookup.GetTypeFromScalarType(scalarType.Value);
                tensorData = PythonDataHelper.Parse(stringValues, returnType);

                if (tensorData is not Array && 
                    !ScalarTypeLookup.Types.Contains(tensorData.GetType()))
                    throw new ArgumentException("Invalid tensor data type.");

                validatedStringValues = PythonDataHelper.Format(tensorData);

                return true;
            }

            catch
            {
                returnType = null;
                tensorData = null;
                validatedStringValues = null;

                return false;
            }
        }

        private Expression BuildTensorFromArray(Array arrayValues, Type returnType)
        {
            var rank = arrayValues.Rank;
            int[] lengths = [.. Enumerable.Range(0, rank).Select(arrayValues.GetLength)];

            var arrayCreationExpression = Expression.NewArrayBounds(returnType, [.. lengths.Select(len => Expression.Constant(len))]);
            var arrayVariable = Expression.Variable(arrayCreationExpression.Type, "array");
            var assignArray = Expression.Assign(arrayVariable, arrayCreationExpression);

            List<Expression> assignments = [];
            for (int i = 0; i < _stringValues.Length; i++)
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
                [arrayVariable],
                assignArray,
                Expression.Block(assignments),
                arrayVariable
            );

            var tensorCreationMethodInfo = typeof(torch).GetMethod(
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
                Expression.Constant(_scalarType, typeof(ScalarType?)),
                Expression.Constant(_device, typeof(Device)),
                Expression.Constant(false, typeof(bool)),
                Expression.Constant(null, typeof(string).MakeArrayType())
            );

            var tensorVariable = Expression.Variable(typeof(torch.Tensor), "tensor");
            var assignTensor = Expression.Assign(tensorVariable, tensorAssignment);

            var buildTensor = Expression.Block(
                [tensorVariable],
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
                [valueVariable],
                assignValue,
                valueVariable
            );

            var tensorCreationMethodInfo = typeof(torch).GetMethod(
                "tensor", [
                    returnType,
                    typeof(Device),
                    typeof(bool)
                ]
            );

            Expression[] tensorCreationMethodArguments = [
                Expression.Constant(_device, typeof(Device)),
                Expression.Constant(false, typeof(bool))
            ];

            if (tensorCreationMethodInfo == null)
            {
                tensorCreationMethodInfo = typeof(torch).GetMethod(
                    "tensor", [
                        returnType,
                        typeof(ScalarType?),
                        typeof(Device),
                        typeof(bool)
                    ]
                );

                tensorCreationMethodArguments = [.. tensorCreationMethodArguments.Prepend(
                    Expression.Constant(_scalarType, typeof(ScalarType?))
                )];
            }

            tensorCreationMethodArguments = [.. tensorCreationMethodArguments.Prepend(
                tensorDataInitializationBlock
            )];

            var tensorAssignment = Expression.Call(
                tensorCreationMethodInfo,
                tensorCreationMethodArguments
            );

            var tensorVariable = Expression.Variable(typeof(torch.Tensor), "tensor");
            var assignTensor = Expression.Assign(tensorVariable, tensorAssignment);

            var buildTensor = Expression.Block(
                [tensorVariable],
                assignTensor,
                tensorVariable
            );

            return buildTensor;
        }

        /// <inheritdoc/>
        public override Expression Build(IEnumerable<Expression> arguments)
        {
            Type[] argTypes = [.. arguments.Select(arg => arg.Type)];

            Type[] methodInfoArgumentTypes = [typeof(Tensor)];

            MethodInfo[] methods = [.. typeof(CreateTensor).GetMethods(BindingFlags.Public | BindingFlags.Instance).Where(m => m.Name == "Process")];

            var methodInfo = arguments.Count() > 0 ? methods.FirstOrDefault(m => m.IsGenericMethod)
                .MakeGenericMethod(
                    arguments
                        .First()
                        .Type
                        .GetGenericArguments()[0]
                ) : methods.FirstOrDefault(m => !m.IsGenericMethod);

            var buildTensor = _tensorData is Array arrayValues ? BuildTensorFromArray(arrayValues, _returnType) : BuildTensorFromScalarValue(_tensorData, _returnType);
            var methodArguments = arguments.Count() == 0 ? [buildTensor] : arguments.Concat([buildTensor]);

            return Expression.Call(
                Expression.Constant(this),
                methodInfo,
                methodArguments
            );
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
            return source.Select(_ => tensor);
        }
    }
}
