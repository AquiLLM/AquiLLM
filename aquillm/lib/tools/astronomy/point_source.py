"""Point-source detection on 2D image arrays (DAOStarFinder)."""


def detect_sources_csv_bytes(data) -> tuple[int, bytes]:
    """
    Run DAOStarFinder on ``data`` and return (source_count, utf-8 CSV bytes).

    Raises on failure (caller maps to tool exception).
    """
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder
    import io

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
    sources = daofind(data - median)

    if sources is None or len(sources) == 0:
        return 0, b""

    df = sources.to_pandas()
    csv_io = io.StringIO()
    df.to_csv(csv_io, index=False)
    return len(df), csv_io.getvalue().encode("utf-8")


__all__ = ["detect_sources_csv_bytes"]
