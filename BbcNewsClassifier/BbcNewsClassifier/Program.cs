using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace BbcNewsClassifier
{
    class Program
    {
        static string _testSet = "news-test.txt";
        static string _trainingSet = "news-train.txt";
        static string _modelPath = "model.txt";
        static List<string> _categories = new List<string> { "business", "entertainment", "politics", "sport", "tech" };

        static Random _random = new Random();

        static void Main(string[] args)
        {
            PredictionModel<NewsData, NewsPrediction> model = null;
            if (File.Exists(_modelPath))
            {
                model = PredictionModel.ReadAsync<NewsData, NewsPrediction>(_modelPath).Result;
            }

            if (model == null)
            {
                PrepareData();
                model = Train();
                model.WriteAsync(_modelPath).Wait();
            }

            Evaluate(model);

            while (true)
            {
                Console.WriteLine();
                Console.WriteLine("Input text: ");
                var text = Console.ReadLine();

                if (text == "Exit")
                {
                    return;
                }

                var prediction = model.Predict(new NewsData { Text = text });

                Console.WriteLine("Prediction result:");
                for (var i = 0; i < prediction.Score.Count(); i++)
                {
                    Console.WriteLine($"{_categories[i]}: {prediction.Score[i]:P2}");
                }
            }
        }

        private static void PrepareData()
        {
            File.Delete(_trainingSet);
            File.Delete(_testSet);

            var basePath = "bbc/";

            var training = new List<NewsData>();
            var test = new List<NewsData>();

            for (var i = 0; i < _categories.Count(); i++)
            {
                var category = _categories[i];
                var path = basePath + category + "/";
                var files = Directory.GetFiles(path);

                var texts = new List<string>();
                foreach (var file in files)
                {
                    var text = File.ReadAllText(file);

                    var textParts = text.Split("\n").ToList();
                    textParts.RemoveAll(s => string.IsNullOrEmpty(s));
                    text = textParts[0] + " " + textParts[1];

                    text = text.Replace(Environment.NewLine, " ");
                    text = text.Replace("\n", " ");
                    text = text.Replace("\r", " ");
                    text = text.Replace("   ", " ");

                    texts.Add(text);
                }

                texts = texts.OrderBy(s => _random.Next()).ToList();

                var trainingTextsCount = (texts.Count / 100) * 80;
                var trainingTexts = texts.GetRange(0, trainingTextsCount);
                training.AddRange(trainingTexts.Select(s => new NewsData { Text = s, Label = category }).ToList());

                var testTexts = texts.GetRange(trainingTextsCount, texts.Count - trainingTextsCount);
                test.AddRange(testTexts.Select(s => new NewsData { Text = s, Label = category }).ToList());
            }

            File.AppendAllLines(_testSet, test.Select(s => $"{s.Text}\t{s.Label}"));
            File.AppendAllLines(_trainingSet, training.Select(s => $"{s.Text}\t{s.Label}"));
        }

        private static PredictionModel<NewsData, NewsPrediction> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<NewsData>(_trainingSet, useHeader: false, separator: "tab"));
            pipeline.Add(new TextFeaturizer("Features", "Text")
            {
                KeepDiacritics = false,
                KeepPunctuations = false,
                TextCase = TextNormalizerTransformCaseNormalizationMode.Lower,
                OutputTokens = true,
                Language = TextTransformLanguage.English,
                StopWordsRemover = new PredefinedStopWordsRemover(),
                VectorNormalizer = TextTransformTextNormKind.L2,
                CharFeatureExtractor = new NGramNgramExtractor() { NgramLength = 3, AllLengths = false },
                WordFeatureExtractor = new NGramNgramExtractor() { NgramLength = 3, AllLengths = true }
            });
            pipeline.Add(new Dictionarizer("Label"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            return pipeline.Train<NewsData, NewsPrediction>();
        }

        public static void Evaluate(PredictionModel<NewsData, NewsPrediction> model)
        {
            var testData = new TextLoader<NewsData>(_trainingSet, useHeader: false, separator: "tab");
            var evaluator = new ClassificationEvaluator();
            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"AccuracyMacro: {metrics.AccuracyMacro:P2}");
            Console.WriteLine($"AccuracyMicro: {metrics.AccuracyMicro:P2}");
            Console.WriteLine($"LogLoss: {metrics.LogLoss:P2}");
        }
    }
}
