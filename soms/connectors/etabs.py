import comtypes.client;
import sys;

class ETABS:
    """Class to call ETABS using CSi API and extract results or construct and run the model.
    """

    def __init__(self):
        self._client

    @property
    def client(self):
        return self._client
    
    @client.setter
    def client(self, client):
        self._client = client

    @classmethod
    def from_instance(self):
        """Connect to running ETABS Instance of ETABS

        Returns
        -------
        cOAPI Pointer
            EtabsObject
        """
        # Attach a running instance of ETABS
        try:
            EtabsObject=comtypes.client.GetActiveObject("CSI.ETABS.API.ETABSObject")
        except (OSError,comtypes.COMError):
            print("No running instance of the program found or failed to attach.")
            sys.exit(-1)

        self.client = EtabsObject
            
        return

    @classmethod
    def from_path(self, path):
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
        
        try:
            EtabsObject=comtypes.client.GetActiveObject("CSI.ETABS.API.ETABSObject")
        except (OSError,comtypes.COMError):
            print("No running instance of the program found or failed to attach.")
            sys.exit(-1)

        self.client = EtabsObject
            
        return
    
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


