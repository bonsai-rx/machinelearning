using System;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Reflection;
using Bonsai;
using Bonsai.Editor;
using Newtonsoft.Json;
using Bonsai.ML.LinearDynamicalSystems;

namespace Bonsai.ML.Examples.Tests.ReceptiveFieldSimpleCell;

[TestClass]
public class ComparingBonsaiPythonOutput
{
    private string basePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ReceptiveFieldSimpleCell");
    private int nSamples = 10000;

    private void RunProcess(string fileName, string fmtArg)
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

        using (var process = new Process {StartInfo = start})
        {
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

    private void DownloadData(string basePath)
    {
        string zipFileUrl = "https://zenodo.org/records/10879253/files/ReceptiveFieldSimpleCell.zip";
        string outputPath = Path.Combine(basePath, "ReceptiveFieldSimpleCell");
        string tempFilePath = Path.Combine(Path.GetTempPath(), "tempfile.zip");

        try
        {
            using (var httpClient = new HttpClient())
            {
                var responseBytes = httpClient.GetByteArrayAsync(zipFileUrl).Result;
                File.WriteAllBytes(tempFilePath, responseBytes);
                Console.WriteLine("File downloaded successfully.");
            }

            ZipFile.ExtractToDirectory(tempFilePath, outputPath, true);
            Console.WriteLine("File extracted successfully.");
        }

        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
        }

        finally
        {
            if (File.Exists(tempFilePath))
            {
                File.Delete(tempFilePath);
                Console.WriteLine("Temporary file deleted.");
            }
        }
    }

    private void RunPythonScript(string basePath)
    {
        var pythonExec = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? "python"
            : "python3";

        RunProcess(pythonExec, $"\"{basePath}/run_python_test.py\" {basePath} {nSamples}");

        Console.WriteLine("Run python script finished.");
    }

    private void RunBonsaiWorkflow(string basePath)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            RunProcess("powershell", $"\"{basePath}\\run_bonsai_test.ps1\" {basePath} {nSamples}");
        }

        else
        {
            RunProcess("/bin/bash", $"\"{basePath}/run_bonsai_test.sh\" {basePath} {nSamples}");
        }
        
        Console.WriteLine("Run bonsai workflow finished.");
    }

    private string GetJsonData(string jsonFileName)
    {
        string jsonString = File.ReadAllText(jsonFileName);
        State state = JsonConvert.DeserializeObject<State>(jsonString);
        string jsonState = JsonConvert.SerializeObject(state);
        return jsonState;
    }

    private State GetStateFromJson(string jsonFileName)
    {
        string jsonString = File.ReadAllText(jsonFileName);
        State state = JsonConvert.DeserializeObject<State>(jsonString);
        return state;
    }

    private bool CompareJSONData(string basePath)
    {
        var bonsaiFileName = Path.Combine(basePath, "bonsai-receptivefield.json");
        var pythonFileName = Path.Combine(basePath, "python-receptivefield.json");

        var bonsaiOutput = GetStateFromJson(bonsaiFileName);
        var pythonOutput = GetStateFromJson(pythonFileName);

        var tolerance = 1e-9;

        try
        {
            for (int i = 0; i < bonsaiOutput.X.GetLength(0); i++)
            {
                for (int j = 0; j < bonsaiOutput.X.GetLength(1); j++)
                {
                    if (Math.Abs(bonsaiOutput.X[i,j] - pythonOutput.X[i,j]) > tolerance) return false;
                }
            }
            for (int i = 0; i < bonsaiOutput.P.GetLength(0); i++)
            {
                for (int j = 0; j < bonsaiOutput.P.GetLength(1); j++)
                {
                    if (Math.Abs(bonsaiOutput.P[i,j] - pythonOutput.P[i,j]) > tolerance) return false;
                }
            }
        }
        catch
        {
            return false;
        }
        return true;
    }

    [TestInitialize]
    [DeploymentItem("run_python_test.py")]
    [DeploymentItem("receptive_field.py")]
    [DeploymentItem("run_bonsai_test.sh")]
    [DeploymentItem("run_bonsai_test.ps1")]
    [DeploymentItem("receptive_field.bonsai")]
    [DeploymentItem("Bonsai.config")]
    [DeploymentItem("NuGet.config")]
    public void TestSetup()
    {
        Directory.CreateDirectory(basePath);
        DownloadData(basePath);
        RunPythonScript(basePath);
        RunBonsaiWorkflow(basePath);
    }

    [TestMethod]
    public void CompareResults()
    {
        var result = CompareJSONData(basePath);
        Assert.IsTrue(result);
    }

    [TestCleanup]
    public void Cleanup()
    {
        if (Directory.Exists(basePath))
        {
            Directory.Delete(basePath, true);
        }
    }
}
