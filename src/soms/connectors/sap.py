import comtypes.client;
import sys;

class SAP:
    """Class to call SAP using CSi API and extract results or construct and run the model.
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
        """Connect to running SAP Instance of ETABS

        Returns
        -------
        cOAPI Pointer
            SAPObject
        """
        # Attach a running instance of ETABS
        try:
            EtabsObject=comtypes.client.GetActiveObject("CSI.ETABS.API.ETABSObject")
        except (OSError,comtypes.COMError):
            print("No running instance of the program found or failed to attach.")
            sys.exit(-1)

        self.client = EtabsObject.SapModel
            
        return
