using Newtonsoft.Json;
using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Bonsai.ML.Tests.Utilities;

namespace Bonsai.ML.Torch.LDS.Tests;

/// <summary>
/// Tests for the neural latents workflow.
/// </summary>
[TestClass]
public class NeuralLatentsTest
{
    private readonly string basePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory);

    private static void RunPythonScript(string basePath)
    {
        var pythonExec = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? "python"
            : "python3";
        var scriptPath = Path.Combine(basePath, "bootstrap_test_environment.py");
        ProcessHelper.RunProcess(pythonExec, $"\"{scriptPath}\" {basePath}");

        Console.WriteLine("Run python script finished.");
    }

    private static async Task RunBonsaiWorkflow(string basePath)
    {
        var currentDirectory = Environment.CurrentDirectory;
        Environment.CurrentDirectory = basePath;
        try
        {
            var workflowPath = Path.Combine(basePath, "NeuralLatentsTest.bonsai");
            await WorkflowHelper.RunWorkflow(
                workflowPath);
            Console.WriteLine("Run bonsai workflow finished.");
        }
        finally { Environment.CurrentDirectory = currentDirectory; }
    }
    
    private static double[] ReadBinaryFile(string fileName)
    {
        using var fileStream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
        using var binaryReader = new BinaryReader(fileStream);
        var fileLength = fileStream.Length;
        var numDoubles = fileLength / sizeof(double);
        var data = new double[numDoubles];
        for (int i = 0; i < numDoubles; i++)
        {
            data[i] = binaryReader.ReadDouble();
        }
        return data;
    }

    private static bool CompareBinaryData(string basePath, double tolerance = 1e-4)
    {
        var bonsaiMeansFileName = Path.Combine(basePath, "bonsai_means.bin");
        var bonsaiCovariancesFileName = Path.Combine(basePath, "bonsai_covs.bin");

        var pythonMeansFileName = Path.Combine(basePath, "python_means.bin");
        var pythonCovariancesFileName = Path.Combine(basePath, "python_covs.bin");

        var bonsaiMeans = ReadBinaryFile(bonsaiMeansFileName);
        var bonsaiCovariances = ReadBinaryFile(bonsaiCovariancesFileName);
        var pythonMeans = ReadBinaryFile(pythonMeansFileName);
        var pythonCovariances = ReadBinaryFile(pythonCovariancesFileName);
        
        if (bonsaiMeans.Length != pythonMeans.Length ||
            bonsaiCovariances.Length != pythonCovariances.Length)
        {
            return false;
        }

        for (int i = 0; i < bonsaiMeans.Length; i++)
        {
            if (Math.Abs(bonsaiMeans[i] - pythonMeans[i]) > tolerance)
            {
                return false;
            }
        }

        for (int i = 0; i < bonsaiCovariances.Length; i++)
        {
            if (Math.Abs(bonsaiCovariances[i] - pythonCovariances[i]) > tolerance)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Setup for the test.
    /// </summary>
    [TestInitialize]
    [DeploymentItem("bootstrap_test_environment.py")]
    [DeploymentItem("estimate_neural_latents.py")]
    [DeploymentItem("NeuralLatentsTest.bonsai")]
    public async Task TestSetup()
    {
        Directory.CreateDirectory(basePath);
        RunPythonScript(basePath);
        await RunBonsaiWorkflow(basePath);
    }

    /// <summary>
    /// Compares the results from the Python script and the Bonsai workflow.
    /// </summary>
    [TestMethod]
    public void CompareResults()
    {
        var result = CompareBinaryData(basePath);
        Assert.IsTrue(result);
    }
}
