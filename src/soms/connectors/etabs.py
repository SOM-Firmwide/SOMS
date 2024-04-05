import comtypes.client
import sys
from soms.datastructures import Frame
from soms.datastructures import Joint
from soms.datastructures import Area


class ETABS:
    """Class to call ETABS using CSi API and extract results or construct and run the model.
    """

    def __init__(self):
        self._client = None
        self._model = None
        self.frames = {}

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, client):
        self._client = client

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @classmethod
    def from_instance(cls):
        """Connect to running ETABS Instance of ETABS

        Returns
        -------
        cOAPI Pointer
            EtabsObject
        """

        session = cls()

        try:
            EtabsObject = comtypes.client.GetActiveObject("CSI.ETABS.API.ETABSObject")
        except (OSError, comtypes.COMError):
            print("No running instance of the program found or failed to attach.")
            sys.exit(-1)

        session.client = EtabsObject
        session.model = EtabsObject.SapModel

        return session

    @classmethod
    def from_path(cls, path):
        """Connect to a specific path ETABS Instance of ETABS

        Parameters
        ----------
        path : str
            Path with the saved ETABS

        Returns
        -------
        cOAPI Pointer
            EtabsObject
        """

        return NotImplementedError("Command in development.")

    def disconnect(self, close=True):
        """Disconnect form the running instance

        Parameters
        ----------
        close : bool, optional
            Whether or not should close the program, by default True
        """

        if close:
            self.client.ApplicationExit(False)

        self.client = None

    def run(self):
        """Run ETABS Client

        Returns
        -------
        None
            Instance runs
        """
        if not self.client:
            return ValueError("ETABS Client not initialized")

        self.client.Analyze.RunAnalysis()

    def define_material(self):
        """Define Material

        Returns
        -------
        None
            Instance runs
        """
        if not self.client:
            return ValueError("ETABS Client not initialized")

        self.client.PropMaterial.SetMaterial()

    def GetFrames(self):
        if not self.client:
            return ValueError("ETABS Client not initialized")

        ret = self.model.FrameObj.GetAllFrames()
        NumberNames, MyName, PropName, StoryName, PointName1, PointName2, Point1X, Point1Y, Point1Z, Point2X, Point2Y, \
            Point2Z, Angle, Offset1X, Offset2X, Offset1Y, Offset2Y, Offset1Z, Offset2Z, CardinalPoint, csys = ret

        dict = {}
        key_name = {}
        key = 0
        for i in range(NumberNames):
            p1 = Joint(Point1X[i], Point1Y[i], Point1Z[i], name=PointName1[i])
            p2 = Joint(Point2X[i], Point2Y[i], Point2Z[i], name=PointName2[i])
            frame = Frame(p1, p2, section=PropName[i], name=MyName[i])
            frame.StoryName = StoryName[i]
            frame.Angle = Angle[i]
            frame.Offset = [Offset1X, Offset2X, Offset1Y, Offset2Y, Offset1Z, Offset2Z]
            frame.CardinalPoint = CardinalPoint[i]

            dict[key] = frame
            key_name[key] = MyName[i]
            key += 1

        self.frames = dict
        self.frames_key_name = key_name

        return dict

    def GetJoints(self):
        if not self.client:
            return ValueError("ETABS Client not initialized. Check")

        ret = self.model.PointObj.GetAllPoints()
        NumberNames, MyName, X, Y, Z, cys = ret

        dict = {}
        key_name = {}
        key = 0
        for i in range(NumberNames):
            ret, _ = self.model.PointObj.GetRestraint(MyName[i])
            pt = Joint(X[i], Y[i], Z[i], name=MyName[i])
            pt.restraint = ret
            dict[key] = pt
            key_name[key] = MyName[i]
            key += 1

        self.joints = dict
        self.joints_key_name = key_name

        return dict

    def GetAreas(self):
        if not self.client:
            return ValueError("ETABS Client not initialized")

        ret = self.model.AreaObj.GetAllAreas()
        NumberNames, MyName, DesignOrientation, NumberBoundaryPts, PointDelimiter, \
            PointNames, PointX, PointY, PointZ, _ = ret

        dict = {}
        key_name = {}
        key = 0
        for i in range(NumberNames):
            if i == 0:
                a, b = 0, PointDelimiter[0] + 1
            else:
                a, b = PointDelimiter[i-1] + 1, PointDelimiter[i] + 1
            pts = []
            for j in range(len(PointNames[a:b])):
                pt = Joint(PointX[a:b][j], PointY[a:b][j], PointZ[a:b][j], name=PointNames[a:b][j])
                pts.append(pt)
            areai = Area(points=pts, name=MyName[i])
            dict[key] = areai
            key_name[key] = MyName[i]
            key += 1

        self.areas = dict
        self.areas_key_name = key_name

        return dict
