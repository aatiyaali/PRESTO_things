import astropy.units as u



class NonEquivalentError(Exception):
    pass



def uconv(value, unit):
    """
    Performs unit conversion between a value and a desired unit. If the value
    is unitless, the desired unit will be assigned to it. If the value has
    existing units that are compatible with the desired unit, the conversion
    will be performed.

    Parameters
    ----------
    value : scalar or `~astropy.units.Unit`
        The quantity to check for units. If scalar, units will assumed to be
        the same as in the "unit" argument.
    unit : `~astropy.units.Unit`
        The unit to check against. If the "quantity" argument already has an
        associated unit, a conversion will be attempted.
    """

    if unit.is_equivalent(value):
            return value.to(unit)

    elif hasattr(value, 'unit'):
        raise NonEquivalentError("Non-equivalent units")

    else:
        return value * unit
        warnings.warn("Assuming value is in {}".format(unit))
