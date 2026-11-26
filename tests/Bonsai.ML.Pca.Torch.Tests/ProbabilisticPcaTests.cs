using System.Diagnostics;
using Bonsai.ML.Pca.Torch;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch.Tests;

/// <summary>
/// Tests for the probabilistic PCA model fitted with the EM algorithm.
/// </summary>
[TestClass]
public class ProbabilisticPcaTests
{
    private readonly ulong _seed = 0;
    private readonly float[] _scales = [4f, 2f, 0.5f, 0.4f, 0.3f];

    public ProbabilisticPcaTests()
    {
        manual_seed(0);
        set_printoptions(style: TorchSharp.TensorStringStyle.Numpy);
    }

    private ProbabilisticPca CreateModel(int numComponents = 2)
    {
        return new ProbabilisticPca
        {
            NumComponents = numComponents,
            Iterations = 2000,
            Tolerance = 1e-6,
            Generator = new Generator(_seed)
        };
    }

    [TestMethod]
    public void TestMatchesClosedFormSolution()
    {
        var (data, _, _) = OnlinePcaTestHelpers.GenerateDataset(5000, _scales);
        var model = CreateModel();
        model.Fit(data);

        var reference = new Pca { NumComponents = 5 };
        reference.Fit(data);
        var eigenvalues = reference.SingularValues.pow(2) / data.size(0);

        // The maximum likelihood variance is the mean of the discarded eigenvalues
        var expectedVariance = eigenvalues.slice(0, 2, 5, 1).mean().item<float>();
        Debug.WriteLine($"Variance: {model.Variance}, expected: {expectedVariance}");
        Assert.AreEqual(expectedVariance, model.Variance, 0.01 * expectedVariance);

        // The components are the principal axes scaled by sqrt(eigenvalue - variance)
        var gram = model.Components.T.matmul(model.Components);
        Debug.WriteLine($"Gram matrix: {gram.str()}");
        for (int i = 0; i < 2; i++)
        {
            var similarity = OnlinePcaTestHelpers.CosineSimilarity(model.Components.select(1, i), reference.Components.select(1, i));
            Assert.IsTrue(similarity > 0.999, $"Similarity of component {i} was {similarity}");

            var expectedStrength = eigenvalues[i].item<float>() - expectedVariance;
            Assert.AreEqual(expectedStrength, gram[i, i].item<float>(), 0.02 * expectedStrength);
        }
        Assert.IsTrue(gram[0, 1].abs().item<float>() < 1e-3);
    }

    [TestMethod]
    public void TestLogLikelihoodDoesNotDecrease()
    {
        var (data, _, _) = OnlinePcaTestHelpers.GenerateDataset(2000, _scales);
        var model = CreateModel();
        model.Type = ScalarType.Float64;
        model.Fit(data);

        var logLikelihood = model.LogLikelihood;
        var count = logLikelihood.size(0);
        Assert.IsTrue(count > 1 && count < model.Iterations, $"Number of iterations was {count}");
        Assert.IsTrue(logLikelihood.isfinite().all().item<bool>());

        var increments = logLikelihood.slice(0, 1, count, 1) - logLikelihood.slice(0, 0, count - 1, 1);
        var smallestIncrement = increments.min().item<double>();
        Debug.WriteLine($"Iterations: {count}, smallest log likelihood increment: {smallestIncrement}");
        Assert.IsTrue(smallestIncrement > -1e-6, $"Smallest increment was {smallestIncrement}");
    }

    [TestMethod]
    public void TestReconstructionMatchesStandardPca()
    {
        var (data, _, _) = OnlinePcaTestHelpers.GenerateDataset(5000, _scales, offsets: [5f, -3f, 1f, 0f, 2f]);
        var model = CreateModel();
        model.Fit(data);

        var reference = new Pca { NumComponents = 2 };
        reference.Fit(data);

        var reconstructed = model.Reconstruct(model.Transform(data));
        var expected = reference.Reconstruct(reference.Transform(data));
        var maxDifference = (reconstructed - expected).abs().max().item<float>();
        Debug.WriteLine($"Largest difference from the standard PCA reconstruction: {maxDifference}");
        Assert.IsTrue(maxDifference < 1e-2, $"Reconstructions differ by {maxDifference}");
    }

    [TestMethod]
    public void TestRefitWithDevice()
    {
        var (data, _, _) = OnlinePcaTestHelpers.GenerateDataset(500, _scales);
        var model = CreateModel();
        model.Device = CPU;

        model.Fit(data);
        var firstVariance = model.Variance;
        model.Generator = new Generator(_seed);
        model.Fit(data);

        Assert.AreEqual(firstVariance, model.Variance, 1e-6);
        Assert.AreEqual(2, model.Transform(data).size(1));
    }
}
