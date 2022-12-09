class one_phase_config:
    """This class contains the configuration parameters of the class
    """
    
    def __init__(self, phase_conf):
        self._name = phase_conf['name']
        self._samples_count = phase_conf['selection_criteria']['samples_count']
        self._time_duration = phase_conf['selection_criteria']['time_duration']

class phases_config:
    """This class is an array of one_phas_conf
    """
    def __init__(self, phases_conf):
        self._phases_conf = []
        
        for phase_conf in phases_conf['phase_config']:
            self._phases_conf.append(one_phase_config(phase_conf))
        
        