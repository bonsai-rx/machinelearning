using System.Diagnostics;
using Bonsai.ML.Pca.Torch;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch.Tests;

/// <summary>
/// Tests for the Bonsai.ML.Pca.Torch package.
/// </summary>
[TestClass]
public class StandardPcaTests
{
    public StandardPcaTests()
    {
        manual_seed(0);
        set_printoptions(style: TorchSharp.TensorStringStyle.Numpy);
    }

    private static readonly Tensor _expectedFirstComponent = tensor(new float[] { 1f, 0f });

    private static Tensor Generate2dRotationMatrix(double angleDegrees)
    {
        var angleRad = angleDegrees * Math.PI / 180.0;
        var cosA = Math.Cos(angleRad);
        var sinA = Math.Sin(angleRad);
        return tensor(new float[,]
        {
            { (float)cosA, (float)-sinA },
            { (float)sinA, (float)cosA }
        });
    }

    private static Tensor Generate2dDataset(int numSamples, double scaleX = 3.0, double scaleY = 0.1, double offsetX = 0.0, double offsetY = 0.0, double rotationAngle = 0.0)
    {
        var x = randn(numSamples) * scaleX + offsetX;
        var y = randn(numSamples) * scaleY + offsetY;
        var data = stack([x, y], 1);
        if (rotationAngle == 0)
            return data;
        var rotationMatrix = Generate2dRotationMatrix(rotationAngle);
        return data.matmul(rotationMatrix);
    }

    private static float Similarity(Tensor a, Tensor b)
    {
        if (a.dim() != 1 || b.dim() != 1)
            throw new ArgumentException("Input tensors must be 1-dimensional. Instead got dimension: " + a.dim());
        if (a.size(0) != b.size(0))
            throw new ArgumentException($"Input tensors must have the same shape. Instead got {string.Join(", ", a.shape)} and {string.Join(", ", b.shape)}.");
        var aNorm = a.norm(-1, keepdim: true);
        var bNorm = b.norm(-1, keepdim: true);
        var dotProduct = a.dot(b);
        return abs(dotProduct / (aNorm * bNorm)).item<float>();
    }

    private static void TestBasic(PcaBaseModel model)
    {
        // Generate a simple dataset.
        var data = Generate2dDataset(1000);
        Debug.WriteLine($"Data shape: {string.Join(", ", data.shape)}");

        // Fit the model.
        model.Fit(data);

        // Verify components.
        Debug.WriteLine($"Components: {model.Components.str()}");
        Assert.IsTrue(model.Components.shape[0] == 2 && model.Components.shape[1] == 2);

        // Verify similarity with expected first component.
        var similarity = Similarity(model.Components[0], _expectedFirstComponent);
        Debug.WriteLine($"Similarity with expected first component: {similarity}");
        Assert.IsTrue(similarity > 0.99);

        // Compare reconstructed data.
        var transformed = model.Transform(data);
        var reconstructed = model.Reconstruct(transformed);

        var reconstructionError = mean((data - reconstructed).pow(2)).item<float>();
        Debug.WriteLine($"Reconstruction error: {reconstructionError}");
        Assert.IsTrue(reconstructionError < 1e-10);
    }

    private static void TestRotation(PcaBaseModel model)
    {
        // Compare rotated dataset
        var data = Generate2dDataset(1000, rotationAngle: 30.0);
        model.Fit(data);

        var rotationMatrix = Generate2dRotationMatrix(30.0);
        var rotatedExpectedFirstComponent = _expectedFirstComponent.matmul(rotationMatrix);
        Debug.WriteLine($"Rotated expected first component: {rotatedExpectedFirstComponent.str()}");
        var rotatedSimilarity = Similarity(model.Components[0], rotatedExpectedFirstComponent);
        Debug.WriteLine($"Similarity with expected first component (rotated data): {rotatedSimilarity}");
        Assert.IsTrue(rotatedSimilarity > 0.99);
    }

    private static void TestOffset(PcaBaseModel model)
    {
        // Test offset centering
        var dataOffset = Generate2dDataset(1000, offsetX: 5.0, offsetY: -3.0);
        model.Fit(dataOffset);

        var offsetExpectedFirstComponent = _expectedFirstComponent;
        var offsetSimilarity = Similarity(model.Components[0], offsetExpectedFirstComponent);
        Debug.WriteLine($"Similarity with expected first component (offset data): {offsetSimilarity}");
        Assert.IsTrue(offsetSimilarity > 0.99);

        var transformed = model.Transform(dataOffset);
        var reconstructed = model.Reconstruct(transformed);

        var reconstructedMeans = reconstructed.mean([0]);
        Debug.WriteLine($"Reconstructed means (offset data): {reconstructedMeans.str()}");
        Assert.IsTrue(abs(reconstructedMeans[0] - 5.0).item<float>() < 0.1);
        Assert.IsTrue(abs(reconstructedMeans[1] + 3.0).item<float>() < 0.1);
    }

    [TestMethod]
    public void TestStandardPca()
    {
        var pca = new Pca(numComponents: 2);
        TestBasic(pca);

        pca = new Pca(numComponents: 2);
        TestRotation(pca);

        pca = new Pca(numComponents: 2);
        TestOffset(pca);
    }
}
