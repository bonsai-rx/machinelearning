using Newtonsoft.Json;
using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Bonsai.ML.Tests.Utilities;
using static TorchSharp.torch;
using TorchSharp;

namespace Bonsai.ML.Lds.Torch.Tests;

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

    private static double[] ReadBinaryFile(string fileName)
    {
        Console.WriteLine($"Reading binary file: {fileName}");
        using var fileStream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
        using var binaryReader = new BinaryReader(fileStream);
        var fileLength = fileStream.Length;
        var numDoubles = fileLength / sizeof(double);
        var data = new double[numDoubles];
        for (int i = 0; i < numDoubles; i++)
        {
            data[i] = binaryReader.ReadDouble();
        }
        Console.WriteLine($"Read {numDoubles} doubles from {fileName}");
        return data;
    }

    private static void WriteToTensor(string fileName, long[] shape)
    {
        Console.WriteLine($"Reading filename: {fileName} and creating tensor with shape [{string.Join(", ", shape)}]");
        var data = ReadBinaryFile(fileName);
        var tensor = from_array(data).reshape(shape);
        var outputFileName = Path.ChangeExtension(fileName, ".pt");
        tensor.Save(outputFileName);
        Console.WriteLine($"Saved tensor to {outputFileName}");
    }

    private static void ConvertBinaryFiles(string basePath)
    {
        var transformedBinnedSpikesFileName = Path.Combine(basePath, "transformed_binned_spikes.bin");
        WriteToTensor(transformedBinnedSpikesFileName, [142, -1]);

        var transitionMatrixFileName = Path.Combine(basePath, "python_B0.bin");
        WriteToTensor(transitionMatrixFileName, [10, 10]);

        var measurementFunctionFileName = Path.Combine(basePath, "python_Z0.bin");
        WriteToTensor(measurementFunctionFileName, [142, 10]);

        var processNoiseFileName = Path.Combine(basePath, "python_Q0.bin");
        WriteToTensor(processNoiseFileName, [10, 10]);

        var observationNoiseFileName = Path.Combine(basePath, "python_R0.bin");
        WriteToTensor(observationNoiseFileName, [142, 142]);

        var initialStateFileName = Path.Combine(basePath, "python_m0_0.bin");
        WriteToTensor(initialStateFileName, [10]);

        var initialCovarianceFileName = Path.Combine(basePath, "python_V0_0.bin");
        WriteToTensor(initialCovarianceFileName, [10, 10]);

        var outputMeansFileName = Path.Combine(basePath, "python_means.bin");
        WriteToTensor(outputMeansFileName, [10, -1]);

        var outputCovariancesFileName = Path.Combine(basePath, "python_covs.bin");
        WriteToTensor(outputCovariancesFileName, [10, 10, -1]);
    }

    private static async Task RunBonsaiWorkflow(string basePath)
    {
        Console.WriteLine($"Running Bonsai workflow...");
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
        ConvertBinaryFiles(basePath);
        await RunBonsaiWorkflow(basePath);
    }

    /// <summary>
    /// Cleanup files generated for test.
    /// </summary>
    [TestCleanup]
    public void TestCleanup()
    {
        var ptFiles = Directory.GetFiles(basePath, "*.pt");
        var binFiles = Directory.GetFiles(basePath, "*.bin");
        foreach (var file in ptFiles) File.Delete(file);
        foreach (var file in binFiles) File.Delete(file);

        var virtualEnvPath = Path.Combine(basePath, ".venv");
        if (Directory.Exists(virtualEnvPath))
        {
            Directory.Delete(virtualEnvPath, true);
        }

        var remfileCachePath = Path.Combine(basePath, "remfile_cache");
        if (Directory.Exists(remfileCachePath))
        {
            Directory.Delete(remfileCachePath, true);
        }
    }

    /// <summary>
    /// Compares the results from the Python script and the Bonsai workflow.
    /// </summary>
    [TestMethod]
    public void CompareTensorData()
    {
        var bonsaiMeansFileName = Path.Combine(basePath, "bonsai_means.pt");
        var bonsaiCovariancesFileName = Path.Combine(basePath, "bonsai_covs.pt");

        var pythonMeansFileName = Path.Combine(basePath, "python_means.pt");
        var pythonCovariancesFileName = Path.Combine(basePath, "python_covs.pt");

        var bonsaiMeans = Tensor.Load(bonsaiMeansFileName);
        var bonsaiCovariances = Tensor.Load(bonsaiCovariancesFileName);
        var pythonMeans = Tensor.Load(pythonMeansFileName).permute(1, 0);
        var pythonCovariances = Tensor.Load(pythonCovariancesFileName).permute(2, 0, 1);

        Assert.IsTrue(allclose(bonsaiMeans, pythonMeans));
        Assert.IsTrue(allclose(bonsaiCovariances, pythonCovariances));
    }
}
