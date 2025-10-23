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

    private static void DownloadData(string basePath)
    {
        string zipFileUrl = "https://zenodo.org/records/17427805/files/Bonsai.ML.Lds.Torch.Tests.zip";

        try
        {
            byte[] responseBytes;
            using (var httpClient = new HttpClient())
            {
                responseBytes = httpClient.GetByteArrayAsync(zipFileUrl).Result;
                Console.WriteLine("File downloaded successfully.");
            }

            using MemoryStream zipStream = new(responseBytes);
            using ZipArchive zip = new(zipStream, ZipArchiveMode.Read);
            zip.ExtractToDirectory(basePath);
            Console.WriteLine("File extracted successfully.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
        }
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
    [DeploymentItem("NeuralLatentsTest.bonsai")]
    public async Task TestSetup()
    {
        Directory.CreateDirectory(basePath);
        DownloadData(basePath);
        await RunBonsaiWorkflow(basePath);
    }

    /// <summary>
    /// Cleanup files generated for test.
    /// </summary>
    [TestCleanup]
    public void TestCleanup()
    {
        var ptFiles = Directory.GetFiles(basePath, "*.pt");
        var zipFiles = Directory.GetFiles(basePath, "*.zip");
        foreach (var file in ptFiles) File.Delete(file);
        foreach (var file in zipFiles) File.Delete(file);
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
