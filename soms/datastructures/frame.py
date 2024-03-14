"""Frame Class"""


class Frame:
    def __init__(self, start, end, section=None, name=None):
        self.name = name
        
        self._start = None
        self._end = None

        self.start = start
        self.end = end
        self.section = section

    @property
    def start(self):
        return self._start
    
    @start.setter
    def start(self, start):
        self._start = start


    @property
    def end(self):
        return self._end
    
    @end.setter
    def end(self, end):
        self._end = end

    def length(self):
            qd = (self.end[0] - self.start[0])**2 + (self.end[1] - self.start[1])**2 +(self.end[2] - self.start[2])**2
            if qd < 1e-10:
                return ValueError("Length is Zero.")
            return qd**0.5