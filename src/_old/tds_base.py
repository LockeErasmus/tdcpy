"""

TODO:
    1. all necessary verifications
"""

from abc import ABC



class TimeDelaySystem:
    def __init__(self) -> None:
        
        self.E = None
        self.A = None
        self.hA = None
        self.B1 = None
        self.hB1 = None
        self.C1 = None
        self.hC1 = None
        self.D11 = None
        self.hD11 = None
        self.B2 = None
        self.hB2 = None
        self.C2 = None
        self.hC2 = None
        self.D12 = None
        self.hD12 = None
        self.D21 = None
        self.hD21 = None
        self.D22 = None
        self.hD22 = None
    
    @property
    def mA(self) -> int | None:
        return len(self.hA) if self.hA else None

    @property
    def mB1(self) -> int | None:
        return len(self.hB1) if self.hB1 else None
    
    @property
    def mC1(self) -> int | None:
        return len(self.hC1) if self.hC1 else None
    
    @property
    def mD11(self) -> int | None:
        return len(self.hD11) if self.hD11 else None
    
    @property
    def mB2(self) -> int | None:
        return len(self.hB2) if self.hB2 else None
    
    @property
    def mC2(self) -> int | None:
        return len(self.hC2) if self.hC2 else None
    
    @property
    def mD12(self) -> int | None:
        return len(self.hD12) if self.hD12 else None
    
    @property
    def mD21(self) -> int | None:
        return len(self.hD21) if self.hD21 else None
    
    @property
    def mD22(self) -> int | None:
        return len(self.hD22) if self.hD22 else None
    
    @property
    def n(self) -> int | None:
        return len(self.A[0])
    
    # TODO also other dimensions
    
    @property
    def is_real(self) -> bool:
        return True # TODO
    
    @property
    def is_sorted(self) -> bool:
        return True # TODO
    
    def is_compressed(self) -> bool:
        return True
    
    @property
    def is_lti(self) -> bool:
        return True # TODO
    
    @property
    def is_logical(self) -> bool:
        return False # TODO
    
    @property
    def is_dde(self) -> bool:
        return False # TODO
    
    

    

    

    

    def compress(self):
        pass

    def sort(self):
        pass
    


    def delay_difference_form(self):
        # HAST TO BE IMPLEMENTED
        pass

    def assymptotic_transfer_function(self):
        pass

