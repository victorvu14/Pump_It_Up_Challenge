# Pump_It_Up_Challenge

The goal of the case study is to create a model able to classify which pumps are functional, which need repairs, and which are non-functional. In addition, the goal is also to understand which variables are the most important in classifying a pump’s functionality. The results obtained will help improve maintenance operations for water pumps and ensure that clean, potable water is available to communities across Tanzania.

# Conclusion 

Multiple models were submitted to the competition. Both the tuned and untuned versions of the RFC and GBT were considered, as well as a One vs Rest (OVR) classifier (which fits a RFC to each different class of the target variable). Finally, a stacked ensemble model, comprised of the best two previous models (tuned RFC and GBT) plus the OVR was fit. The stacked model used the predictions of the previous models as input to classify the target variable on the test set. Ideally the stacked model would be able to benefit from uncorrelated errors of the different models, achieving an even higher accuracy. It also was attempted to fit a model to a reduced set of 46 variables, based on feature importance, as well as using the entire dataset before RFE. The following scores were obtained:


In conclusion, the tuned Random Forest produced the highest accuracy at 82.32%, reaching rank 197 in the competition. The stacked model using tuned RF, tuned GBT and OVR obtained the second highest accuracy at 82.07%. Even though the pure Random Forest performed best in the competition, it might be advisable to proceed with the stacked model instead. It only performed slightly worse, while being significantly more stable. Depending on the random seed set, the RF was achieving scores between 0.8201 and 0.8232, whereas the stacked model was only ranging from 0.8200 to 0.8207.


# Rank: 197/6590
