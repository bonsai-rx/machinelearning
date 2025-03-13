using System;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Bonsai.ML.Data
{
    /// <summary>
    /// Provides a set of static methods for working with JSON data containing numeric and boolean values.
    /// </summary>
    public static class JsonDataHelper
    {
        /// <summary>
        /// Parses the input string into an object of the specified type.
        /// </summary>
        /// <param name="input">The string to parse.</param>
        /// <param name="dtype">The data type of the object.</param>
        /// <returns>An object of the specified type containing the parsed data.</returns>
        public static object Parse(string input, Type dtype)
        {
            if (!IsValidJson(input))
            {
                throw new ArgumentException($"Parameter: {nameof(input)} is not valid JSON.");
            }

            var token = JsonConvert.DeserializeObject<JToken>(input);

            if (token is JValue value)
            {
                return Convert.ChangeType(value, dtype);
            }

            var output = ParseToken(token, dtype);

            return output;
        }

        private static bool IsValidJson(string input)
        {
            try
            {
                JToken.Parse(input);
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Parses the input token into an object of the specified type. 
        /// If the input is a JSON array, the method will attempt to parse it into a list or array of the specified type.
        /// </summary>
        /// <param name="token">The token to parse.</param>
        /// <param name="dtype">The data type of the object.</param>
        /// <returns>An object of the specified type containing the parsed data.</returns>
        public static object ParseToken(JToken token, Type dtype)
        {
            if (token is JValue value)
            {
                return Convert.ChangeType(value, dtype);
            }
            else if (token is JArray)
            {
                if (token[0] is JValue)
                {
                    if (token.All(item => item is JValue))
                    {
                        int depth = ParseDepth(token);
                        return ParseArray(token, dtype, depth);
                    }
                    return CreateList(token, dtype);
                }
                else
                {
                    var subArrayDimensions = token.Cast<JArray>().Select(value =>
                    {
                        var depth = ParseDepth(value);
                        return ParseDimensions(value, depth);
                    }).ToList();

                    if (subArrayDimensions.All(s => s.SequenceEqual(subArrayDimensions[0])))
                    {
                        return ParseArray(token, dtype, subArrayDimensions[0].Count());
                    }
                    return CreateList(token, dtype);
                }
            }
            else
            {
                throw new ArgumentException($"Error parsing parameter: {nameof(token)}. JSON input is not supported.");
            }
        }

        private static object ParseArray(JToken token, Type dtype, int depth)
        {
            int[] dimensions = ParseDimensions(token, depth);
            var resultArray = Array.CreateInstance(dtype, dimensions);
            PopulateArray(token, resultArray, [], dtype);
            return resultArray;
        }

        private static int ParseDepth(JToken token, int currentDepth = 0)
        {
            if (token is JArray arr && arr.Count > 0)
            {
                return ParseDepth(arr[0], currentDepth + 1);
            }
            return currentDepth;
        }

        private static int[] ParseDimensions(JToken token, int depth, int currentLevel = 0)
        {
            if (depth == 0 || token is not JArray)
            {
                return [0];
            }

            List<int> dimensions = [];
            var current = token;

            while (current != null && current is JArray currentArray)
            {
                dimensions.Add(currentArray.Count);
                if (currentArray.Count > 0)
                {
                    if (currentArray.Any(item => item is not JArray) && currentArray.Any(item => item is JArray) || currentArray.All(item => item is JArray) && currentArray.Any(item => ((JArray)item).Count != ((JArray)currentArray.First()).Count))
                    {
                        throw new ArgumentException($"Error parsing parameter: {nameof(token)}. Array dimensions are inconsistent.");
                    }

                    if (currentArray.First() is not JArray)
                    {
                        if (!currentArray.All(item => double.TryParse(item.ToString(), out _)) && !currentArray.All(item => bool.TryParse(item.ToString(), out _)))
                        {
                            throw new ArgumentException($"Error parsing parameter: {nameof(token)}. All values in the array must be of the same type. Only numeric or boolean types are supported.");
                        }
                    }
                }
                current = currentArray.Count > 0 ? currentArray[0] : null;
            }

            if (currentLevel > 0 && token is JArray arr && arr.All(x => x is JArray))
            {
                var subArrayDimensions = new HashSet<string>();
                foreach (JArray subArr in arr)
                {
                    int[] subDims = ParseDimensions(subArr, depth - currentLevel, currentLevel + 1);
                    subArrayDimensions.Add(string.Join(",", subDims));
                }

                if (subArrayDimensions.Count > 1)
                {
                    throw new ArgumentException("Inconsistent array dimensions.");
                }
            }

            return [.. dimensions];
        }

        private static void PopulateArray(JToken token, Array array, int[] indices, Type dtype)
        {
            if (token is JArray arr)
            {
                for (int i = 0; i < arr.Count; i++)
                {
                    int[] newIndices = new int[indices.Length + 1];
                    Array.Copy(indices, newIndices, indices.Length);
                    newIndices[newIndices.Length - 1] = i;
                    PopulateArray(arr[i], array, newIndices, dtype);
                }
            }
            else
            {
                var values = Convert.ChangeType(token, dtype);
                array.SetValue(values, indices);
            }
        }

        private static object CreateList(JToken token, Type dtype)
        {
            var listType = typeof(List<>).MakeGenericType(DetermineListType(token, dtype));
            var list = (IList)Activator.CreateInstance(listType);

            foreach (var item in token)
            {
                var result = ParseToken(item, dtype);
                list.Add(result);
            }

            return list;
        }

        private static Type DetermineListType(JToken token, Type type)
        {
            if (token.All(item => item is JValue))
            {
                return type;
            }
            else if (token.All(item => item is JArray))
            {
                var subArrayDepth = token.Cast<JArray>().Select(value => ParseDepth(value)).ToList();

                if (subArrayDepth.All(s => s == subArrayDepth[0]))
                {
                    var rank = subArrayDepth[0];
                    if (rank > 1)
                    {
                        return type.MakeArrayType(rank);
                    }
                    return type.MakeArrayType();
                }
                else
                {
                    return typeof(List<>).MakeGenericType(DetermineListType(token[0], type));
                }
            }
            else
            {
                return typeof(object);
            }
        }

        /// <summary>
        /// Formats the input object into a string representation that is consistent with JSON syntax.
        /// </summary>
        /// <param name="obj">The object to format.</param>
        /// <returns>A string representation that is consistent with JSON syntax.</returns>
        public static string Format(object obj)
        {
            var sb = new StringBuilder();
            int depth = 0;
            Format(obj, sb, depth);
            return sb.ToString();
        }

        private static void Format(object obj, StringBuilder sb, int depth)
        {
            switch (obj)
            {
                case null:
                    sb.Append("null");
                    break;
                case string:
                case char:
                    sb.Append('"').Append(obj).Append('"');
                    break;
                case bool:
                    sb.Append(obj.ToString().ToLower());
                    break;
                case int:
                case double:
                case float:
                case long:
                case short:
                case byte:
                case ushort:
                case uint:
                case ulong:
                case sbyte:
                case decimal:
                    sb.Append(obj);
                    break;
                case Array:
                    FormatArray(obj, sb, depth);
                    break;
                case IList:
                    FormatList(obj, sb, depth);
                    break;
                case IDictionary:
                    FormatDictionary(obj, sb, depth);
                    break;
                case object tuple when obj.GetType().IsGenericType && 
                        obj.GetType().GetGenericTypeDefinition() == typeof(Tuple<,>) || 
                        obj.GetType().GetGenericTypeDefinition() == typeof(Tuple<,,>) ||
                        obj.GetType().GetGenericTypeDefinition() == typeof(Tuple<,,,>) ||
                        obj.GetType().GetGenericTypeDefinition() == typeof(Tuple<,,,,>) ||
                        obj.GetType().GetGenericTypeDefinition() == typeof(Tuple<,,,,,>) ||
                        obj.GetType().GetGenericTypeDefinition() == typeof(Tuple<,,,,,,>) ||
                        obj.GetType().GetGenericTypeDefinition() == typeof(Tuple<,,,,,,,>):
                    FormatTuple(obj, sb, depth);
                    break;
                default:
                    FormatDefault(obj, sb, depth);
                    break;
            }
        }

        private static void FormatArray(object obj, StringBuilder sb, int depth)
        {
            var array = (Array)obj;
            if (array.Rank == 1)
            {
                sb.Append('[');
                for (int i = 0; i < array.Length; i++)
                {
                    if (i > 0)
                    {
                        sb.Append(", ");
                    }
                    Format(array.GetValue(i), sb, depth + 1);
                }
                sb.Append(']');
            }
            else
            {
                FormatNDArray(array, new int[array.Rank], 0, sb, depth);
            }
        }

        private static void FormatNDArray(Array array, int[] indices, int dimension, StringBuilder sb, int depth)
        {
            if (dimension == array.Rank)
            {
                Format(array.GetValue(indices), sb, depth + 1);
                return;
            }

            sb.Append('[');
            for (int i = 0; i < array.GetLength(dimension); i++)
            {
                if (i > 0)
                {
                    sb.Append(", ");
                }
                indices[dimension] = i;
                FormatNDArray(array, indices, dimension + 1, sb, depth + 1);
            }
            sb.Append(']');
        }

        private static void FormatList(object obj, StringBuilder sb, int depth)
        {
            var list = (IList)obj;
            sb.Append('[');
            for (int i = 0; i < list.Count; i++)
            {
                if (i > 0)
                {
                    sb.Append(", ");
                }
                Format(list[i], sb, depth + 1);
            }
            sb.Append(']');
        }

        private static void FormatDictionary(object obj, StringBuilder sb, int depth)
        {
            var dict = (IDictionary)obj;
            sb.Append('{');
            bool first = true;
            foreach (DictionaryEntry entry in dict)
            {
                if (!first)
                {
                    sb.Append(", ");
                }
                else
                {
                    first = false;
                }
                Format(entry.Key, sb, depth + 1);
                sb.Append(": ");
                Format(entry.Value, sb, depth + 1);
            }
            sb.Append('}');
        }

        private static void FormatTuple(object obj, StringBuilder sb, int depth)
        {
            var itemProperties = obj.GetType()
                .GetProperties()
                .Where(p => p.Name.StartsWith("Item"))
                .OrderBy(p => p.Name)
                .ToArray();

            sb.Append('(');
            for (int i = 0; i < itemProperties.Length; i++)
            {
                if (i > 0)
                {
                    sb.Append(", ");
                }
                Format(itemProperties[i].GetValue(obj), sb, depth + 1);
            }
            sb.Append(')');
        }

        private static void FormatDefault(object obj, StringBuilder sb, int depth)
        {

            var properties = obj.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance);
            sb.Append('{');
            for (int i = 0; i < properties.Length; i++)
            {
                if (i > 0)
                {
                    sb.Append(", ");
                }
                sb.Append('"').Append(properties[i].Name).Append("\": ");
                Format(properties[i].GetValue(obj), sb, depth + 1);
            }
            sb.Append('}');
        }
    }
}