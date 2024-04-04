"""Joint Class"""

from compas.geometry import Point


class Joint(Point):
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
        self._restraint = None
        self.name = name
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

    @z.setter
    def z(self, z):
        self._z = float(z)

    @property
    def restraint(self):
        return self._restraint

    @restraint.setter
    def restraint(self, value):
        try:
            iter(value)
            if len(value) == 3:
                self._restraint = [value[0], value[1], value[2], False, False, False]
            elif len(value) == 6:
                self._restraint = value
            else:
                raise ValueError("Please provide an restraint interable with length 3 or 6")
        except Exception:
            raise ValueError("Please provide an restraint interable with length 3 or 6")
