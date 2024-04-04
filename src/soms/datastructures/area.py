"""Area Class"""

from compas.geometry import Polygon

class Area(Polygon):
    def __init__(self, points=None, name=None):
        self.name = name
        self._points = None

        self.points = points

    @property
    def points(self):
        return self._points
    
    @points.setter
    def points(self, points):
        self._points = points
