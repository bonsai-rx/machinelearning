using System;
using System.Linq;
using System.Reactive.Linq;
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
                var rangeParts = indexString.Split(':');
                rangeParts = [.. rangeParts.Where(p => {
                    p = p.Trim();
                    return !string.IsNullOrEmpty(p);
                })];

                if (rangeParts.Length == 0)
                {
                    indices[i] = torch.TensorIndex.Slice();
                }
                else if (rangeParts.Length == 1)
                {
                    indices[i] = torch.TensorIndex.Slice(int.Parse(rangeParts[0]));
                }
                else if (rangeParts.Length == 2)
                {
                    indices[i] = torch.TensorIndex.Slice(int.Parse(rangeParts[0]), int.Parse(rangeParts[1]));
                }
                else if (rangeParts.Length == 3)
                {
                    indices[i] = torch.TensorIndex.Slice(int.Parse(rangeParts[0]), int.Parse(rangeParts[1]), int.Parse(rangeParts[2]));
                }
                else
                {
                    throw new Exception($"Invalid index format: {indexString}");
                }
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