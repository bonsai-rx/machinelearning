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
        private readonly Dictionary<Type, Action<object, StringBuilder>> typeHandlers;
        private readonly Dictionary<Type, PropertyInfo[]> typeProperties;
        private readonly StringBuilder sb;

        /// <summary>
        /// Initializes a new instance of the <see cref="StringFormatter"/> class.
        /// </summary>
        public StringFormatter()
        {
            typeHandlers = new Dictionary<Type, Action<object, StringBuilder>>();
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
            ConvertCSharpToPythonStringInternal(obj, sb);
            return sb.ToString();
        }

        private void ConvertCSharpToPythonStringInternal(object obj, StringBuilder sb)
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

            handler(obj, sb);
        }

        private Action<object, StringBuilder> CreateTypeHandler(Type type)
        {
            if (type == typeof(string) || type == typeof(char))
            {
                return (obj, sb) => sb.Append('"').Append(obj).Append('"');
            }

            if (type == typeof(bool))
            {
                return (obj, sb) => sb.Append(((bool)obj).ToString().ToLower());
            }

            if (type == typeof(int) || type == typeof(double) || type == typeof(float) || type == typeof(long) || type == typeof(short) || type == typeof(byte) || type == typeof(ushort) || type == typeof(uint) || type == typeof(ulong) || type == typeof(sbyte) || type == typeof(decimal))
            {
                return (obj, sb) => sb.Append(obj);
            }

            if (type.IsArray)
            {
                return (obj, sb) =>
                {
                    var array = (Array)obj;
                    sb.Append('[');
                    for (int i = 0; i < array.Length; i++)
                    {
                        if (i > 0) sb.Append(", ");
                        ConvertCSharpToPythonStringInternal(array.GetValue(i), sb);
                    }
                    sb.Append(']');
                };
            }

            if (typeof(IList).IsAssignableFrom(type))
            {
                return (obj, sb) =>
                {
                    var list = (IList)obj;
                    sb.Append('[');
                    for (int i = 0; i < list.Count; i++)
                    {
                        if (i > 0) sb.Append(", ");
                        ConvertCSharpToPythonStringInternal(list[i], sb);
                    }
                    sb.Append(']');
                };
            }

            if (typeof(IDictionary).IsAssignableFrom(type))
            {
                return (obj, sb) =>
                {
                    var dict = (IDictionary)obj;
                    sb.Append('{');
                    bool first = true;
                    foreach (DictionaryEntry entry in dict)
                    {
                        if (!first) sb.Append(", ");
                        ConvertCSharpToPythonStringInternal(entry.Key, sb);
                        sb.Append(": ");
                        ConvertCSharpToPythonStringInternal(entry.Value, sb);
                        first = false;
                    }
                    sb.Append('}');
                };
            }

            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Tuple<,>))
            {
                var itemProperties = type.GetProperties().Where(p => p.Name.StartsWith("Item")).OrderBy(p => p.Name).ToArray();
                return (obj, sb) =>
                {
                    sb.Append('(');
                    for (int i = 0; i < itemProperties.Length; i++)
                    {
                        if (i > 0) sb.Append(", ");
                        ConvertCSharpToPythonStringInternal(itemProperties[i].GetValue(obj), sb);
                    }
                    sb.Append(')');
                };
            }

            if (!typeProperties.TryGetValue(type, out var properties))
            {
                properties = type.GetProperties(BindingFlags.Public | BindingFlags.Instance);
                typeProperties[type] = properties;
            }

            return (obj, sb) =>
            {
                sb.Append('{');
                for (int i = 0; i < properties.Length; i++)
                {
                    if (i > 0) sb.Append(", ");
                    sb.Append('"').Append(properties[i].Name).Append("\": ");
                    ConvertCSharpToPythonStringInternal(properties[i].GetValue(obj), sb);
                }
                sb.Append('}');
            };
        }
    }
}