null_or_numeric = lambda df, col_name : pd.to_numeric(df[col_name], errors='coerce').notnull().all()
def is_integer(val):
    try: 
        return val == int(val)
    except (ValueError, TypeError) as error:
        # Returning false if can't coerce to an int.
        return False

def is_float(val):
    try: 
        return val == float(val)
    except (ValueError, TypeError) as error:
        # Returning false if can't coerce to a float.
        return False

is_integer_in_range = lambda val, min, max: is_integer(val) and (int(val) <= max and int(val) >= min)
is_float_in_range = lambda val, min, max: is_float(val) and (float(val) <= max and float(val) >= min)

class UnitConverter:
    __cf_to_af = 0.0000229569
    __af_to_gallons = 325851
    
    def cf_to_af(self, val):
        return val * self.__cf_to_af

    def af_to_cf(self, val):
        return val / self.__cf_to_af
    
    def gallons_to_af(self, val):
        return val / self.__af_to_gallons
        
    def af_to_gallons(self, val):
        return val * self.__af_to_gallons
    
unit_converter = UnitConverter()
    