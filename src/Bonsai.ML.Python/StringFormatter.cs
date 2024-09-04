using System;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;

namespace Bonsai.ML.Python
{
    /// <summary>
    /// Represents a C# to Python string formatter class.
    /// </summary>
    public class StringFormatter
    {
        private readonly Dictionary<Type, Action<object, StringBuilder, int>> typeHandlers;
        private readonly Dictionary<Type, PropertyInfo[]> typeProperties;
        private readonly StringBuilder sb;

        public StringFormatter()
        {
            typeHandlers = new Dictionary<Type, Action<object, StringBuilder, int>>();
            typeProperties = new Dictionary<Type, PropertyInfo[]>();
            sb = new StringBuilder();
        }

        /// <summary>
        /// Formats the specified object into a string that is consistent with Python syntax.
        /// </summary>
        /// <param name="obj">The object to format.</param>
        /// <returns>A string that is consistent with Python syntax.</returns>
        public string Format(object obj)
        {
            sb.Clear();
            ConvertCSharpToPythonStringInternal(obj, sb, 0);
            return sb.ToString();
        }

        private void ConvertCSharpToPythonStringInternal(object obj, StringBuilder sb, int depth)
        {
            if (obj == null)
            {
                sb.Append("None");
                return;
            }

            var type = obj.GetType();

            if (!typeHandlers.TryGetValue(type, out var handler))
            {
                handler = CreateTypeHandler(type);
                typeHandlers[type] = handler;
            }

            handler(obj, sb, depth);
        }

        private Action<object, StringBuilder, int> CreateTypeHandler(Type type)
        {
            if (type == typeof(string) || type == typeof(char))
                return (obj, sb, _) => sb.Append('"').Append(obj).Append('"');

            if (type == typeof(bool))
                return (obj, sb, _) => sb.Append(((bool)obj).ToString().ToLower());

            if (type == typeof(int) || type == typeof(double) || type == typeof(float) || type == typeof(long) || type == typeof(short) || type == typeof(byte) || type == typeof(ushort) || type == typeof(uint) || type == typeof(ulong) || type == typeof(sbyte) || type == typeof(decimal))
                return (obj, sb, _) => sb.Append(obj);

            if (type.IsArray)
                return CreateArrayHandler(type);

            if (typeof(IList).IsAssignableFrom(type))
                return CreateListHandler(type);

            if (typeof(IDictionary).IsAssignableFrom(type))
                return CreateDictionaryHandler(type);

            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Tuple<,>))
                return CreateTupleHandler(type);

            return CreateCustomObjectHandler(type);
        }

        private Action<object, StringBuilder, int> CreateArrayHandler(Type type)
        {
            return (obj, sb, depth) =>
            {
                var array = (Array)obj;
                sb.Append('[');
                for (int i = 0; i < array.Length; i++)
                {
                    if (i > 0) sb.Append(", ");
                    ConvertCSharpToPythonStringInternal(array.GetValue(i), sb, depth + 1);
                }
                sb.Append(']');
            };
        }

        private Action<object, StringBuilder, int> CreateListHandler(Type type)
        {
            return (obj, sb, depth) =>
            {
                var list = (IList)obj;
                sb.Append('[');
                for (int i = 0; i < list.Count; i++)
                {
                    if (i > 0) sb.Append(", ");
                    ConvertCSharpToPythonStringInternal(list[i], sb, depth + 1);
                }
                sb.Append(']');
            };
        }

        private Action<object, StringBuilder, int> CreateDictionaryHandler(Type type)
        {
            return (obj, sb, depth) =>
            {
                var dict = (IDictionary)obj;
                sb.Append('{');
                bool first = true;
                foreach (DictionaryEntry entry in dict)
                {
                    if (!first) sb.Append(", ");
                    ConvertCSharpToPythonStringInternal(entry.Key, sb, depth + 1);
                    sb.Append(": ");
                    ConvertCSharpToPythonStringInternal(entry.Value, sb, depth + 1);
                    first = false;
                }
                sb.Append('}');
            };
        }

        private Action<object, StringBuilder, int> CreateTupleHandler(Type type)
        {
            var itemProperties = type.GetProperties().Where(p => p.Name.StartsWith("Item")).OrderBy(p => p.Name).ToArray();
            return (obj, sb, depth) =>
            {
                sb.Append('(');
                for (int i = 0; i < itemProperties.Length; i++)
                {
                    if (i > 0) sb.Append(", ");
                    ConvertCSharpToPythonStringInternal(itemProperties[i].GetValue(obj), sb, depth + 1);
                }
                sb.Append(')');
            };
        }

        private Action<object, StringBuilder, int> CreateCustomObjectHandler(Type type)
        {
            if (!typeProperties.TryGetValue(type, out var properties))
            {
                properties = type.GetProperties(BindingFlags.Public | BindingFlags.Instance);
                typeProperties[type] = properties;
            }

            return (obj, sb, depth) =>
            {
                sb.Append('{');
                for (int i = 0; i < properties.Length; i++)
                {
                    if (i > 0) sb.Append(", ");
                    sb.Append('"').Append(properties[i].Name).Append("\": ");
                    ConvertCSharpToPythonStringInternal(properties[i].GetValue(obj), sb, depth + 1);
                }
                sb.Append('}');
            };
        }
    }
}