import arimafd as oa
import pandas as pd
from nab.detectors.base import AnomalyDetector
from nab.detectors.arimafd_constant import ar_order

class ArimaFD4Detector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super(ArimaFD4Detector, self).__init__(*args, **kwargs)

    def handleRecord(self, inputData):
        return []
    
    def run(self):
        headers = self.getHeader()
        
        ts = self.dataSet.data.set_index("timestamp")
        ts = ts.astype({"value": float})
        ad = oa.Anomaly_detection(ts)
        ad.generate_tensor(ar_order=ar_order)
        ts_anomaly = ad.proc_tensor(No_metric=4)
        ts_anomaly = ts_anomaly.reset_index()
        
        ans = self.dataSet.data.copy()
        deleted_ans = ans.copy()[:ar_order]
        ans = ans.copy()[ar_order:]
        deleted_ans["anomaly"] = 0
        ans["anomaly"] = ts_anomaly[0]
        ans = pd.concat([deleted_ans, ans]).reset_index(drop=True)
        ans.columns = headers
        return ans