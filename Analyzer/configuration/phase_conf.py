class phase_config:
    """This class contains the configuration parameters of the class
    """
    
    def __init__(self, phase_conf):
        self._name = phase_conf['name']
        self._samples_count = phase_conf['selection_criteria']['samples_count']
        self._time_duration = phase_conf['selection_criteria']['time_duration']
        self._epsilon = phase_conf['dbscan_parameters']['epsilon']
        self._min_points = phase_conf['dbscan_parameters']['_min_points']

class sequence_config:
    """This class is a list with the phases config of one sequence
    """
    def __init__(self, sequence_conf):
        self._sequence_name = sequence_conf['sequence_name']
        self._phases_conf = []
        
        for phase_conf in sequence_conf['phases_config']:
            self._phases_conf.append(phase_config(phase_conf))
        
        