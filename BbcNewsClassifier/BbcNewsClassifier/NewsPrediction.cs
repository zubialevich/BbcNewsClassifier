using Microsoft.ML.Runtime.Api;

namespace BbcNewsClassifier
{
    public class NewsPrediction
    {
        [ColumnName("Score")]
        public float[] Score;
    }
}
