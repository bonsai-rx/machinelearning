using Newtonsoft.Json;
using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Bonsai.ML.Tests.Utilities;

namespace Bonsai.ML.Lds.Python.Tests;

/// <summary>
/// Tests for the ReceptiveFieldSimpleCell workflow.
/// </summary>
[TestClass]
public class ReceptiveFieldSimpleCellTest
{
    private readonly string basePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory);

    private static void DownloadData(string basePath)
    {
        string zipFileUrl = "https://zenodo.org/records/10879253/files/ReceptiveFieldSimpleCell.zip";
        string outputPath = Path.Combine(basePath, "data");

        try
        {
            byte[] responseBytes;
            using (var httpClient = new HttpClient())
            {
                httpClient.DefaultRequestHeaders.Add("User-Agent", "Bonsai.ML.Tests");
                responseBytes = httpClient.GetByteArrayAsync(zipFileUrl).Result;
                Console.WriteLine("File downloaded successfully.");
            }

            using MemoryStream zipStream = new(responseBytes);
            using ZipArchive zip = new(zipStream, ZipArchiveMode.Read);
            zip.ExtractToDirectory(outputPath);
            Console.WriteLine("File extracted successfully.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
        }
    }

    private static void RunPythonScript(string basePath)
    {
        var pythonExec = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? "python"
            : "python3";
        var scriptPath = Path.Combine(basePath, "bootstrap_test_environment.py");
        ProcessHelper.RunProcess(pythonExec, $"\"{scriptPath}\" {basePath} 10000");

        Console.WriteLine("Run python script finished.");
    }

    private async Task RunBonsaiWorkflow(string basePath)
    {
        var currentDirectory = Environment.CurrentDirectory;
        Environment.CurrentDirectory = basePath;
        try
        {
            var workflowPath = Path.Combine(basePath, "receptive_field.bonsai");
            await WorkflowHelper.RunWorkflow(
                workflowPath,
                properties: [("NSamples", 10000)]);
            Console.WriteLine("Run bonsai workflow finished.");
        }
        finally { Environment.CurrentDirectory = currentDirectory; }
    }

    private static State GetStateFromJson(string jsonFileName)
    {
        string jsonString = File.ReadAllText(jsonFileName);
        State state = JsonConvert.DeserializeObject<State>(jsonString) ?? new State();
        return state;
    }

    private static bool CompareJSONData(string basePath, double tolerance = 1e-9)
    {
        var originalFileName = Path.Combine(basePath, "original-receptivefield.json");
        var bonsaiFileName = Path.Combine(basePath, "bonsai-receptivefield.json");
        var pythonFileName = Path.Combine(basePath, "python-receptivefield.json");

        var originalOutput = GetStateFromJson(originalFileName);
        var bonsaiOutput = GetStateFromJson(bonsaiFileName);
        var pythonOutput = GetStateFromJson(pythonFileName);

        try
        {
            for (int i = 0; i < bonsaiOutput.X.GetLength(0); i++)
            {
                for (int j = 0; j < bonsaiOutput.X.GetLength(1); j++)
                {
                    if (Math.Abs(bonsaiOutput.X[i, j] - pythonOutput.X[i, j]) > tolerance || Math.Abs(originalOutput.X[i, j] - pythonOutput.X[i, j]) > tolerance)
                    {
                        Console.WriteLine($"Discrepency found comparing X at index ({i},{j}) with tolerance {tolerance}: bonsaiOutput = {bonsaiOutput.X[i, j]}, pythonOutput = {pythonOutput.X[i, j]}, originalOutput = {originalOutput.X[i, j]}.");
                        return false;
                    }
                }
            }
            for (int i = 0; i < bonsaiOutput.P.GetLength(0); i++)
            {
                for (int j = 0; j < bonsaiOutput.P.GetLength(1); j++)
                {
                    if (Math.Abs(bonsaiOutput.P[i, j] - pythonOutput.P[i, j]) > tolerance || Math.Abs(originalOutput.P[i, j] - pythonOutput.P[i, j]) > tolerance)
                    {
                        Console.WriteLine($"Discrepency found comparing P at index ({i},{j}) with tolerance {tolerance}: bonsaiOutput = {bonsaiOutput.P[i, j]}, pythonOutput = {pythonOutput.P[i, j]}, originalOutput = {originalOutput.P[i, j]}.");
                        return false;
                    }
                }
            }
        }
        catch
        {
            return false;
        }
        return true;
    }

    /// <summary>
    /// Setup for the test.
    /// </summary>
    [TestInitialize]
    [DeploymentItem("bootstrap_test_environment.py")]
    [DeploymentItem("receptive_field.py")]
    [DeploymentItem("receptive_field.bonsai")]
    [DeploymentItem("original-receptivefield.json")]
    public async Task TestSetup()
    {
        Directory.CreateDirectory(basePath);
        DownloadData(basePath);
        RunPythonScript(basePath);
        await RunBonsaiWorkflow(basePath);
    }

    /// <summary>
    /// Compares the results from the Python script and the Bonsai workflow.
    /// </summary>
    [TestMethod]
    public void CompareResults()
    {
        var result = CompareJSONData(basePath);
        Assert.IsTrue(result);
    }
}
