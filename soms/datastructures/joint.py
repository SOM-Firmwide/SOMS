"""Joint Class"""


class Joint:
    """A joint defined by XYZ positions

    Parameters
    ----------
    x : float
        The X coordinate of the joint.
    y : float
        The Y coordinate of the joint.
    z : float
        The Z coordinate of the joint.

    """

    def __init__(self, x, y, z, name=None):
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0

        self.x = x
        self.y = y
        self.z = z

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, x):
        self._x = float(x)

    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, y):
        self._y = float(y)

    @property
    def z(self):
        return self._z
    
    @y.setter
    def z(self, z):
        self._z = float(z)


