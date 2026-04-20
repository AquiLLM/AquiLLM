"""Flat-field correction on FITS array data."""


def flat_field_correct(science_data, flat_data):
    """Return science_data / flat_data or raise ValueError on invalid input."""
    if science_data.shape != flat_data.shape:
        raise ValueError("Wrong dimensions! Science and flat-field images must have the same shape.")
    if (flat_data == 0).any():
        raise ValueError("Flat field image contains zero values, cannot safely divide.")
    return science_data / flat_data


__all__ = ["flat_field_correct"]
