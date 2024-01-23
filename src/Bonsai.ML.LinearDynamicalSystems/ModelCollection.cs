using System;
using System.Collections.Generic;
using System.ComponentModel;
using YamlDotNet.Core.Tokens;

namespace Bonsai.ML.LinearDynamicalSystems
{
    public class ModelCollection : StringConverter
    {

        private static List<string> modelNames = new List<string>();
        private static Dictionary<string, int> modelNameCounts = new Dictionary<string, int>();

        public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
        {
            return true;
        }

        public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
        {
            if (context != null)
            {
                return new StandardValuesCollection(modelNames);
            }

            return base.GetStandardValues(context);
        }

        public static void AddModelName(string modelName)
        {
            if (modelName != null)
            {
                
                Console.WriteLine($"Adding model: {modelName}");
                if (modelNames.Contains(modelName))
                {
                    modelNameCounts[modelName]++;
                    // Console.WriteLine($"Model name '{modelName}' already exists in the collection.");
                    // Bonsai.Work
                    throw new InvalidOperationException($"Model name '{modelName}' already exists in the collection.");
                }

                else 
                {
                    modelNameCounts[modelName] = 1;
                    modelNames.Add(modelName);
                }
                // Console.WriteLine(modelNameCounts);
            }
        }

        public static void RemoveModelName(string modelName)
        {
            if (modelName != null)
            {
                Console.WriteLine($"Removing model: {modelName}");
                if (!modelNames.Contains(modelName))
                {
                    throw new ArgumentException($"Model name '{modelName}' does not exist in collection.");
                }

                modelNameCounts[modelName]--;
                // modelNames.Remove(modelName);

                if (modelNameCounts[modelName] <= 0)
                {
                    modelNameCounts.Remove(modelName);
                    modelNames.Remove(modelName);
                }
                // Console.WriteLine(modelNameCounts);
            }
        }
    }
}
