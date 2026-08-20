using System.Diagnostics;
using Bonsai.ML.Pca.Torch;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch.Tests;

internal static class OnlinePcaTestHelpers
{
    public static (Tensor data, Tensor trueComponents, Tensor trueMean) GenerateDataset(
        int numSamples, float[] scales, float[]? offsets = null, double rotationAngle2d = 0.0)
    {
        var numFeatures = scales.Length;
        var latent = randn(numSamples, numFeatures) * tensor(scales);

        Tensor basis;
        if (numFeatures == 2)
        {
            var angleRad = rotationAngle2d * Math.PI / 180.0;
            var cosA = (float)Math.Cos(angleRad);
            var sinA = (float)Math.Sin(angleRad);
            basis = tensor(new float[,] { { cosA, -sinA }, { sinA, cosA } });
        }
        else
        {
            basis = linalg.qr(randn(numFeatures, numFeatures), mode: linalg.QRMode.Reduced).Q;
        }

        var data = latent.matmul(basis.T);
        var trueMean = offsets is null ? zeros(numFeatures) : tensor(offsets);
        data += trueMean;
        return (data, basis, trueMean);
    }

    public static float SubspaceSimilarity(Tensor estimated, Tensor expected)
    {
        var qEstimated = linalg.qr(estimated, mode: linalg.QRMode.Reduced).Q;
        var qExpected = linalg.qr(expected, mode: linalg.QRMode.Reduced).Q;
        var singularValues = linalg.svdvals(qEstimated.T.matmul(qExpected));
        return singularValues.min().item<float>();
    }

    public static float CosineSimilarity(Tensor a, Tensor b)
    {
        var dotProduct = a.dot(b);
        return abs(dotProduct / (a.norm() * b.norm()).clamp_min(1e-12)).item<float>();
    }

    public static void FitStreaming(PcaBaseModel model, Tensor data, int batchSize)
    {
        foreach (var batch in data.split(batchSize))
            model.Fit(batch);
    }
}

/// <summary>
/// Convergence tests for the online PCA model based on the Generalized Hebbian Algorithm.
/// </summary>
[TestClass]
public class OnlinePcaGhaTests
{
    private readonly ulong _seed = 0;

    public OnlinePcaGhaTests()
    {
        set_printoptions(style: TorchSharp.TensorStringStyle.Numpy);
    }

    private OnlinePcaGha CreateModel(int numComponents = 2, float learningRate = 0.1f)
    {
        return new OnlinePcaGha
        {
            NumComponents = numComponents,
            LearningRate = learningRate,
            Generator = new Generator(_seed)
        };
    }

    [TestMethod]
    public void TestConvergenceOnStationaryStream()
    {
        var (data, trueComponents, _) = OnlinePcaTestHelpers.GenerateDataset(10000, [3f, 0.1f]);
        var model = CreateModel();

        var batches = data.split(50);
        var similarities = new List<float>();
        for (int i = 0; i < batches.Length; i++)
        {
            model.Fit(batches[i]);
            if (i % 20 == 0 || i == batches.Length - 1)
                similarities.Add(OnlinePcaTestHelpers.CosineSimilarity(model.Components.select(1, 0), trueComponents.select(1, 0)));
        }
        Debug.WriteLine($"Similarity trajectory: {string.Join(", ", similarities)}");

        Assert.IsTrue(similarities[^1] > 0.99, $"Final similarity was {similarities[^1]}");
        Assert.IsTrue(similarities[^1] >= similarities[0], "Similarity did not improve over training.");

        var gram = model.Components.T.matmul(model.Components);
        var orthonormalityError = (gram - eye(2)).abs().max().item<float>();
        Debug.WriteLine($"Orthonormality error: {orthonormalityError}");
        Assert.IsTrue(orthonormalityError < 1e-5);

        Assert.AreEqual(10000, model.SampleCount);
    }

    [TestMethod]
    public void TestStreamingFitAndTransformMatchesTransformAfterTraining()
    {
        var (data, _, _) = OnlinePcaTestHelpers.GenerateDataset(500, [3f, 0.1f]);
        var model = CreateModel();

        Tensor lastStreamed = empty(0);
        foreach (var sample in data.split(1))
            lastStreamed = model.FitAndTransform(sample);

        var numSamples = data.size(0);
        var lastTransformed = model.Transform(data).slice(0, numSamples - 1, numSamples, 1);
        var maxDifference = (lastStreamed - lastTransformed).abs().max().item<float>();

        Debug.WriteLine($"Last streamed: {lastStreamed.str()}");
        Debug.WriteLine($"Last transformed: {lastTransformed.str()}");
        Debug.WriteLine($"Max difference: {maxDifference}");
        Assert.IsTrue(maxDifference < 1e-6, $"Streamed and batch results differ by {maxDifference}");
    }

    [TestMethod]
    public void TestStreamingFitAndTransformMatchesFitSingleAndBatched()
    {
        var (data, _, _) = OnlinePcaTestHelpers.GenerateDataset(500, [3f, 0.1f]);
        var modelFt = CreateModel();

        Tensor lastFt = empty(0);
        foreach (var sample in data.split(1))
            lastFt = modelFt.FitAndTransform(sample);

        var modelS = CreateModel();

        foreach (var sample in data.split(1))
            modelS.Fit(sample);

        var numSamples = data.size(0);
        var lastS = modelS.Transform(data).slice(0, numSamples - 1, numSamples, 1);

        var maxFtSDifference = (lastFt - lastS).abs().max().item<float>();
        Debug.WriteLine($"Last FitAndTransform: {lastFt.str()}");
        Debug.WriteLine($"Last streamed: {lastS.str()}");
        Debug.WriteLine($"FitAndTransform vs streamed difference: {maxFtSDifference}");
        Assert.IsTrue(maxFtSDifference < 1e-6, $"FitAndTransform and streamed results differ by {maxFtSDifference}");

        var modelB = CreateModel();
        modelB.Fit(data);
        var meanDifference = (modelS.Mean - modelB.Mean).abs().max().item<float>();
        Debug.WriteLine($"Streamed vs batch mean difference: {meanDifference}");
        Assert.IsTrue(meanDifference < 1e-4, $"Streamed and batch means differ by {meanDifference}");

        for (int epoch = 0; epoch < 50; epoch++)
            modelB.Fit(data);

        var subspaceSimilarity = OnlinePcaTestHelpers.SubspaceSimilarity(modelB.Components, modelS.Components);
        Debug.WriteLine($"Streamed vs batch subspace similarity: {subspaceSimilarity}");
        Assert.IsTrue(subspaceSimilarity > 0.99, $"Subspace similarity was {subspaceSimilarity}");
    }

    [TestMethod]
    public void TestConvergenceOnRotatedData()
    {
        var (data, trueComponents, _) = OnlinePcaTestHelpers.GenerateDataset(10000, [3f, 0.1f], rotationAngle2d: 30.0);
        var model = CreateModel();
        OnlinePcaTestHelpers.FitStreaming(model, data, batchSize: 50);

        Debug.WriteLine($"Model components: {model.Components.str()}");
        Debug.WriteLine($"Expected first component: {trueComponents.select(1, 0).str()}");
        var similarity = OnlinePcaTestHelpers.CosineSimilarity(model.Components.select(1, 0), trueComponents.select(1, 0));
        Debug.WriteLine($"Similarity with rotated first component: {similarity}");
        Assert.IsTrue(similarity > 0.99, $"Similarity was {similarity}");
    }

    [TestMethod]
    public void TestMeanTrackingWithOffsetData()
    {
        var (data, trueComponents, trueMean) = OnlinePcaTestHelpers.GenerateDataset(10000, [3f, 0.1f], offsets: [5f, -3f]);
        var model = CreateModel();
        OnlinePcaTestHelpers.FitStreaming(model, data, batchSize: 50);

        var meanError = (model.Mean.squeeze() - trueMean).abs().max().item<float>();
        Debug.WriteLine($"Estimated mean: {model.Mean.str()}, error: {meanError}");
        Assert.IsTrue(meanError < 0.2, $"Mean estimate error was {meanError}");

        var similarity = OnlinePcaTestHelpers.CosineSimilarity(model.Components.select(1, 0), trueComponents.select(1, 0));
        Debug.WriteLine($"Similarity with first component on offset data: {similarity}");
        Assert.IsTrue(similarity > 0.99, $"Similarity was {similarity}");

        var reconstructed = model.Reconstruct(model.Transform(data));
        var reconstructedMeans = reconstructed.mean([0]);
        Debug.WriteLine($"Reconstructed means: {reconstructedMeans.str()}");
        Assert.IsTrue(abs(reconstructedMeans[0] - 5.0).item<float>() < 0.2);
        Assert.IsTrue(abs(reconstructedMeans[1] + 3.0).item<float>() < 0.2);
    }

    [TestMethod]
    public void TestHigherDimensionalSubspaceRecovery()
    {
        var (data, trueComponents, _) = OnlinePcaTestHelpers.GenerateDataset(20000, [4f, 2f, 0.5f, 0.2f, 0.1f]);
        var model = CreateModel();
        OnlinePcaTestHelpers.FitStreaming(model, data, batchSize: 50);

        var trueSubspace = trueComponents.slice(1, 0, 2, 1);
        var subspaceSimilarity = OnlinePcaTestHelpers.SubspaceSimilarity(model.Components, trueSubspace);
        Debug.WriteLine($"Subspace similarity with true top-2 subspace: {subspaceSimilarity}");
        Assert.IsTrue(subspaceSimilarity > 0.98, $"Subspace similarity was {subspaceSimilarity}");

        var firstSimilarity = OnlinePcaTestHelpers.CosineSimilarity(model.Components.select(1, 0), trueComponents.select(1, 0));
        var secondSimilarity = OnlinePcaTestHelpers.CosineSimilarity(model.Components.select(1, 1), trueComponents.select(1, 1));
        Debug.WriteLine($"First component similarity: {firstSimilarity}, second component similarity: {secondSimilarity}");
        Assert.IsTrue(firstSimilarity > 0.98, $"First component similarity was {firstSimilarity}");
        Assert.IsTrue(secondSimilarity > 0.98, $"Second component similarity was {secondSimilarity}");

        var reference = new Pca { NumComponents = 2 };
        reference.Fit(data);
        var referenceSimilarity = OnlinePcaTestHelpers.SubspaceSimilarity(model.Components, reference.Components);
        Debug.WriteLine($"Subspace similarity with standard PCA: {referenceSimilarity}");
        Assert.IsTrue(referenceSimilarity > 0.98, $"Similarity with standard PCA was {referenceSimilarity}");

        var reconstructed = model.Reconstruct(model.Transform(data));
        var reconstructionError = mean((data - reconstructed).pow(2)).item<float>();
        var discardedVariance = (0.5f * 0.5f + 0.2f * 0.2f + 0.1f * 0.1f) / 5f;
        Debug.WriteLine($"Reconstruction error: {reconstructionError}, discarded variance: {discardedVariance}");
        Assert.IsTrue(reconstructionError < 3 * discardedVariance, $"Reconstruction error was {reconstructionError}");
    }
}

/// <summary>
/// Convergence tests for the online probabilistic PCA model.
/// </summary>
[TestClass]
public class OnlineProbabilisticPcaTests
{
    private readonly ulong _seed = 0;

    public OnlineProbabilisticPcaTests()
    {
        set_printoptions(style: TorchSharp.TensorStringStyle.Numpy);
    }

    private OnlineProbabilisticPca CreateModel(int numComponents = 1, double? rho = null, double? kappa = null, int? reorthogonalizePeriod = null)
    {
        return new OnlineProbabilisticPca
        {
            NumComponents = numComponents,
            Rho = rho,
            Kappa = kappa,
            Generator = new Generator(_seed),
            ReorthogonalizePeriod = reorthogonalizePeriod
        };
    }

    [TestMethod]
    public void TestConvergenceWithConstantRho()
    {
        var (data, trueComponents, _) = OnlinePcaTestHelpers.GenerateDataset(10000, [3f, 0.1f]);
        var model = CreateModel(rho: 0.1);
        OnlinePcaTestHelpers.FitStreaming(model, data, batchSize: 50);

        var similarity = OnlinePcaTestHelpers.CosineSimilarity(model.Components.select(1, 0), trueComponents.select(1, 0));
        Debug.WriteLine($"Similarity with true first component: {similarity}");
        Assert.IsTrue(similarity > 0.99, $"Similarity was {similarity}");
    }

    [TestMethod]
    public void TestConvergenceWithDecayingLearningRate()
    {
        var (data, trueComponents, _) = OnlinePcaTestHelpers.GenerateDataset(10000, [3f, 0.1f], rotationAngle2d: 30.0);
        var model = CreateModel(kappa: 0.7);
        OnlinePcaTestHelpers.FitStreaming(model, data, batchSize: 50);

        var similarity = OnlinePcaTestHelpers.CosineSimilarity(model.Components.select(1, 0), trueComponents.select(1, 0));
        Debug.WriteLine($"Similarity with true first component (rotated data): {similarity}");
        Assert.IsTrue(similarity > 0.99, $"Similarity was {similarity}");
    }

    [TestMethod]
    public void TestNoiseVarianceEstimate()
    {
        var (data, _, _) = OnlinePcaTestHelpers.GenerateDataset(10000, [3f, 0.1f]);
        var model = CreateModel(kappa: 0.7);
        OnlinePcaTestHelpers.FitStreaming(model, data, batchSize: 50);

        Debug.WriteLine($"Estimated noise variance: {model.Variance}");
        Assert.IsTrue(model.Variance > 0.002 && model.Variance < 0.05, $"Variance estimate was {model.Variance}");
    }

    [TestMethod]
    public void TestMeanTrackingWithOffsetData()
    {
        var (data, trueComponents, trueMean) = OnlinePcaTestHelpers.GenerateDataset(10000, [3f, 0.1f], offsets: [5f, -3f]);
        var model = CreateModel(kappa: 0.7);
        OnlinePcaTestHelpers.FitStreaming(model, data, batchSize: 50);

        var meanError = (model.Means - trueMean).abs().max().item<float>();
        Debug.WriteLine($"Estimated mean: {model.Means.str()}, error: {meanError}");
        Assert.IsTrue(meanError < 0.2, $"Mean estimate error was {meanError}");

        var similarity = OnlinePcaTestHelpers.CosineSimilarity(model.Components.select(1, 0), trueComponents.select(1, 0));
        Debug.WriteLine($"Similarity with first component on offset data: {similarity}");
        Assert.IsTrue(similarity > 0.99, $"Similarity was {similarity}");

        var reconstructed = model.Reconstruct(model.Transform(data));
        var reconstructedMeans = reconstructed.mean([0]);
        Debug.WriteLine($"Reconstructed means: {reconstructedMeans.str()}");
        Assert.IsTrue(abs(reconstructedMeans[0] - 5.0).item<float>() < 0.2);
        Assert.IsTrue(abs(reconstructedMeans[1] + 3.0).item<float>() < 0.2);
    }

    [TestMethod]
    public void TestHigherDimensionalSubspaceRecovery()
    {
        var (data, trueComponents, _) = OnlinePcaTestHelpers.GenerateDataset(20000, [4f, 2f, 0.5f, 0.2f, 0.1f]);
        var model = CreateModel(numComponents: 2, kappa: 0.7, reorthogonalizePeriod: 10);
        OnlinePcaTestHelpers.FitStreaming(model, data, batchSize: 50);

        var trueSubspace = trueComponents.slice(1, 0, 2, 1);
        var subspaceSimilarity = OnlinePcaTestHelpers.SubspaceSimilarity(model.Components, trueSubspace);
        Debug.WriteLine($"Subspace similarity with true top-2 subspace: {subspaceSimilarity}");
        Assert.IsTrue(subspaceSimilarity > 0.98, $"Subspace similarity was {subspaceSimilarity}");

        var reference = new Pca { NumComponents = 2 };
        reference.Fit(data);
        var referenceSimilarity = OnlinePcaTestHelpers.SubspaceSimilarity(model.Components, reference.Components);
        Debug.WriteLine($"Subspace similarity with batch PCA: {referenceSimilarity}");
        Assert.IsTrue(referenceSimilarity > 0.98, $"Similarity with batch PCA was {referenceSimilarity}");
    }

    [TestMethod]
    public void TestParameterValidation()
    {
        var data = randn(50, 2);

        var bothRates = new OnlineProbabilisticPca { NumComponents = 1, Rho = 0.1, Kappa = 0.7 };
        Assert.ThrowsException<InvalidOperationException>(() => bothRates.Fit(data));

        var noRates = new OnlineProbabilisticPca { NumComponents = 1, Rho = null, Kappa = null };
        Assert.ThrowsException<InvalidOperationException>(() => noRates.Fit(data));

        var invalidRho = new OnlineProbabilisticPca { NumComponents = 1, Rho = 1.5 };
        Assert.ThrowsException<InvalidOperationException>(() => invalidRho.Fit(data));

        var invalidKappa = new OnlineProbabilisticPca { NumComponents = 1, Rho = null, Kappa = 0.4 };
        Assert.ThrowsException<InvalidOperationException>(() => invalidKappa.Fit(data));

        var invalidVariance = new OnlineProbabilisticPca { NumComponents = 1, InitialVariance = 0 };
        Assert.ThrowsException<InvalidOperationException>(() => invalidVariance.Fit(data));
    }
}
