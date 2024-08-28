from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def LassoLarCV_VT_SDS(inputs):
    train_features, test_features, train_labels, vt_value = inputs
    # Feature selection
    vt = VarianceThreshold(threshold=vt_value)
    train_features_vt  = vt.fit_transform(train_features)
    test_features_vt = vt.transform(test_features)

    #SDS
    sds = StandardScaler()   # with_mean=False)
    train_features_vt_sds = sds.fit_transform(train_features_vt)
    test_features_vt_sds = sds.transform(test_features_vt)

    # Create regression object
    regr = LassoLarsCV(fit_intercept=False, max_iter=5000)
    regr.fit(train_features_vt_sds, train_labels)

    return  regr.predict(test_features_vt_sds),  len(train_features_vt[0]), vt_value  # Prediction and no_of_dimensions
