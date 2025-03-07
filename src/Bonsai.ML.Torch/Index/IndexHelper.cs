using System;
using System.Collections.Generic;
using TorchSharp;

namespace Bonsai.ML.Torch.Index;

/// <summary>
/// Provides helper methods to parse tensor indexes.
/// </summary>
public static class IndexHelper
{

    /// <summary>
    /// Parses the input string into an array of tensor indexes.
    /// </summary>
    /// <param name="input"></param>
    public static torch.TensorIndex[] Parse(string input)
    {
        if (string.IsNullOrEmpty(input))
        {
            return [0];
        }

        var indexStrings = input.Split(',');
        var indices = new torch.TensorIndex[indexStrings.Length];

        for (int i = 0; i < indexStrings.Length; i++)
        {
            var indexString = indexStrings[i].Trim();
            if (int.TryParse(indexString, out int intIndex))
            {
                indices[i] = torch.TensorIndex.Single(intIndex);
            }
            else if (indexString == ":")
            {
                indices[i] = torch.TensorIndex.Colon;
            }
            else if (indexString == "None")
            {
                indices[i] = torch.TensorIndex.None;
            }
            else if (indexString == "...")
            {
                indices[i] = torch.TensorIndex.Ellipsis;
            }
            else if (indexString.ToLower() == "false" || indexString.ToLower() == "true")
            {
                indices[i] = torch.TensorIndex.Bool(indexString.ToLower() == "true");
            }
            else if (indexString.Contains(":"))
            {
                string[] rangeParts = [.. indexString.Split(':')];
                var argsList = new List<long?>([null, null, null]);
                try
                {
                    for (int j = 0; j < rangeParts.Length; j++)
                    {
                        if (!string.IsNullOrEmpty(rangeParts[j]))
                        {
                            argsList[j] = long.Parse(rangeParts[j]);
                        }
                    }
                }
                catch (Exception)
                {
                    throw new Exception($"Invalid index format: {indexString}");
                }
                indices[i] = torch.TensorIndex.Slice(argsList[0], argsList[1], argsList[2]);
            }
            else
            {
                throw new Exception($"Invalid index format: {indexString}");
            }
        }
        return indices;
    }
    
    /// <summary>
    /// Serializes the input array of tensor indexes into a string representation.
    /// </summary>
    /// <param name="indexes"></param>
    /// <returns></returns>
    public static string Serialize(torch.TensorIndex[] indexes)
    {
        return string.Join(", ", indexes);
    }
}