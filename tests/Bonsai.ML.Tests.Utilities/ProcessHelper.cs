using System;
using System.Diagnostics;

namespace Bonsai.ML.Tests.Utilities;

/// <summary>
/// Helper class to run external processes in unit tests.
/// </summary>
public class ProcessHelper
{
    /// <summary>
    /// Runs an external process with the specified file name and arguments.
    /// </summary>
    /// <param name="fileName"></param>
    /// <param name="fmtArg"></param>
    public static void RunProcess(string fileName, string fmtArg)
    {
        var start = new ProcessStartInfo
        {
            FileName = fileName,
            Arguments = fmtArg,
            RedirectStandardOutput = true,
            RedirectStandardInput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        using var process = new Process { StartInfo = start };
        process.Start();
        var output = process.StandardOutput.ReadToEnd();
        var error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (!string.IsNullOrEmpty(output))
        {
            Console.WriteLine("Standard Output: ");
            Console.WriteLine(output);
        }

        if (!string.IsNullOrEmpty(error))
        {
            Console.WriteLine("Standard Error: ");
            Console.WriteLine(error);
        }
    }
}