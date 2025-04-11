using System;
using System.Collections.Generic;

namespace Bonsai.ML.Data
{
    /// <summary>
    /// Provides a set of static methods for working with Python data.
    /// </summary>
    public static class PythonDataHelper
    {
        private static readonly Dictionary<string, string> PythonStringConversions = new()
        {
            { "None", "null" },
            { "True", "true" },
            { "False", "false" },
            { "nan", "NaN" },
            { "inf", "Infinity" },
            { "-inf", "-Infinity" }
        };

        private static string ReplacePythonStrings(string value)
        {
            foreach (var conversion in PythonStringConversions)
            {
                value = value.Replace(conversion.Key, conversion.Value);
            }

            return value;
        }

        private static string ReplaceJsonStrings(string value)
        {
            foreach (var conversion in PythonStringConversions)
            {
                value = value.Replace(conversion.Value, conversion.Key);
            }

            return value;
        }

        /// <summary>
        /// Parses the input string into an object of the specified type.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="type"></param>
        /// <returns>An object of the specified type.</returns>
        public static object Parse(string value, Type type)
        {
            return JsonDataHelper.Parse(ReplacePythonStrings(value), type);
        }

        /// <summary>
        /// Formats the specified object into a string that is consistent with Python syntax.
        /// </summary>
        /// <param name="obj"></param>
        /// <returns>A string that is consistent with Python syntax.</returns>
        public static string Format(object obj)
        {
            var result = JsonDataHelper.Format(obj);
            return ReplaceJsonStrings(result);
        }
    }
}