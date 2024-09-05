using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Bonsai.ML.Data
{
    /// <summary>
    /// Provides a set of static methods for working with arrays.
    /// </summary>
    public static class ArrayHelper
    {
        /// <summary>
        /// Serializes the input data into a JSON string representation.
        /// </summary>
        /// <param name="data">The data to serialize.</param>
        /// <returns>A JSON string representation of the input data.</returns>
        public static string SerializeToJson(object data)
        {
            if (data is Array array)
            {
                return SerializeArrayToJson(array);
            }
            else
            {
                return JsonConvert.SerializeObject(data);
            }
        }

        /// <summary>
        /// Serializes the input array into a JSON string representation.
        /// </summary>
        /// <param name="array">The array to serialize.</param>
        /// <returns>A JSON string representation of the input array.</returns>
        public static string SerializeArrayToJson(Array array)
        {
            StringBuilder sb = new StringBuilder();
            SerializeArrayRecursive(array, sb, [0]);
            return sb.ToString();
        }

        private static void SerializeArrayRecursive(Array array, StringBuilder sb, int[] indices)
        {
            if (indices.Length < array.Rank)
            {
                sb.Append("[");
                int length = array.GetLength(indices.Length);
                for (int i = 0; i < length; i++)
                {
                    int[] newIndices = new int[indices.Length + 1];
                    indices.CopyTo(newIndices, 0);
                    newIndices[indices.Length] = i;
                    SerializeArrayRecursive(array, sb, newIndices);
                    if (i < length - 1)
                    {
                        sb.Append(", ");
                    }
                }
                sb.Append("]");
            }
            else
            {
                object value = array.GetValue(indices);
                sb.Append(value.ToString());
            }
        }

        private static bool IsValidJson(string input)
        {
            int squareBrackets = 0;
            foreach (char c in input)
            {
                if (c == '[') squareBrackets++;
                else if (c == ']') squareBrackets--;
            }
            return squareBrackets == 0;
        }

        /// <summary>
        /// Parses the input JSON string into an object of the specified type. If the input is a JSON array, the method will attempt to parse it into an array of the specified type. 
        /// </summary>
        /// <param name="input">The JSON string to parse.</param>
        /// <param name="dtype">The data type of the object.</param>
        /// <returns>An object of the specified type containing the parsed JSON data.</returns>
        public static object ParseString(string input, Type dtype = null)
        {
            if (!IsValidJson(input))
            {
                throw new ArgumentException($"Parameter: {nameof(input)} is not valid JSON.");
            }
            var obj = JsonConvert.DeserializeObject<JToken>(input);
            int depth = ParseDepth(obj);
            if (depth == 0)
            {
                return Convert.ChangeType(input, dtype);
            }
            int[] dimensions = ParseDimensions(obj, depth);
            var resultArray = Array.CreateInstance(dtype, dimensions);
            PopulateArray(obj, resultArray, [0], dtype);
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
            if (depth == 0 || !(token is JArray))
            {
                return [0];
            }

            List<int> dimensions = new List<int>();
            JToken current = token;

            while (current != null && current is JArray)
            {
                JArray currentArray = current as JArray;
                dimensions.Add(currentArray.Count);
                if (currentArray.Count > 0)
                {
                    if (currentArray.Any(item => !(item is JArray)) && currentArray.Any(item => item is JArray) || currentArray.All(item => item is JArray) && currentArray.Any(item => ((JArray)item).Count != ((JArray)currentArray.First()).Count))
                    {
                        throw new ArgumentException($"Error parsing parameter: {nameof(token)}. Array dimensions are inconsistent.");
                    }

                    if (!(currentArray.First() is JArray))
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

            return dimensions.ToArray();
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
    }
}