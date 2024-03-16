"""Structure Class"""


class Structure:
    def __init__(self, name=None):
        self.name = name
        
        self.frames = {}
        self.joints = {}
        self.areas = {}
