using Bonsai.ML.Pca.Torch;
using static TorchSharp.torch;

namespace Bonsai.ML.Pca.Torch.Tests;

/// <summary>
/// Tests for the input handling shared by all PCA models.
/// </summary>
[TestClass]
public class ModelInputTests
{
    private static PcaBaseModel[] CreateModels(ScalarType type)
    {
        return
        [
            new Pca { Type = type },
            new ProbabilisticPca { Type = type },
            new OnlinePcaGha { Type = type },
            new OnlineProbabilisticPca { Type = type }
        ];
    }

    [TestMethod]
    [DataRow(ScalarType.Float32, ScalarType.Float64)]
    [DataRow(ScalarType.Float64, ScalarType.Float32)]
    [DataRow(ScalarType.Float64, ScalarType.Float64)]
    public void TestInputIsConvertedToModelType(ScalarType modelType, ScalarType dataType)
    {
        manual_seed(0);
        var data = randn(200, 4, dtype: dataType);

        foreach (var model in CreateModels(modelType))
        {
            var transformed = model.FitAndTransform(data);
            Assert.AreEqual(modelType, model.Components.dtype, model.GetType().Name);
            Assert.AreEqual(modelType, transformed.dtype, model.GetType().Name);
            CollectionAssert.AreEqual(new long[] { 200, 2 }, transformed.shape, model.GetType().Name);

            var reconstructed = model.Reconstruct(transformed.to_type(dataType));
            Assert.AreEqual(modelType, reconstructed.dtype, model.GetType().Name);
            CollectionAssert.AreEqual(new long[] { 200, 4 }, reconstructed.shape, model.GetType().Name);
        }
    }

    [TestMethod]
    public void TestOutputsSurviveDisposeScope()
    {
        manual_seed(0);
        var data = randn(200, 4);

        foreach (var model in CreateModels(ScalarType.Float32))
        {
            Tensor transformed;
            using (var scope = NewDisposeScope())
            {
                model.Fit(data);
                transformed = model.Transform(data).MoveToOuterDisposeScope();
            }

            Assert.IsFalse(model.Components.IsInvalid, model.GetType().Name);
            Assert.IsFalse(transformed.IsInvalid, model.GetType().Name);
            Assert.AreEqual(4, model.Reconstruct(transformed).size(1), model.GetType().Name);
        }
    }

    [TestMethod]
    public void TestDisposeResetsModel()
    {
        manual_seed(0);
        var data = randn(200, 4);

        foreach (var model in CreateModels(ScalarType.Float32))
        {
            model.Fit(data);
            model.Dispose();
            Assert.AreEqual(-1, model.NumFeatures, model.GetType().Name);
            Assert.ThrowsException<InvalidOperationException>(() => model.Transform(data), model.GetType().Name);

            model.Fit(randn(200, 3));
            Assert.AreEqual(3, model.NumFeatures, model.GetType().Name);
        }
    }

    [TestMethod]
    public void TestStandardPcaRejectsMoreComponentsThanSamples()
    {
        var model = new Pca { NumComponents = 3 };
        Assert.ThrowsException<ArgumentException>(() => model.Fit(randn(2, 4)));
    }
}
