"""Sky subtraction on FITS array data (NumPy / Astropy-friendly)."""


def subtract_sky_arrays(object_data, sky_data):
    """Return object_data - sky_data or raise ValueError if shapes differ."""
    if object_data.shape != sky_data.shape:
        raise ValueError("Wrong dimensions! The object and sky files must have the same dimensions.")
    return object_data - sky_data


__all__ = ["subtract_sky_arrays"]
