import json

class MetricsLogger():
    def __init__(self):
        self._metrics = {}

    def setMetrics(self, metricsSede, metricsMorfo):
        self._metrics['sede'] = metricsSede
        self._metrics['morfo'] = metricsMorfo

    def printMetrics(self, filename):
        with open(filename, 'w') as handler:
            handler.write(json.dumps(self._metrics, sort_keys=True, indent=4))
