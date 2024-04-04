"""Structure Class"""

from compas.datastructures import Assembly


class Structure(Assembly):
    def __init__(self, name=None):
        self.name = name

        self.frames = {}
        self.joints = {}
        self.areas = {}
