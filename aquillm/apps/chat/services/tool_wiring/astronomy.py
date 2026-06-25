"""Astronomy FITS LLM tools (Django-bound)."""
from __future__ import annotations

import io

from django.core.files.base import ContentFile

from aquillm.llm import LLMTool, ToolResultDict, llm_tool
from apps.chat.models import ConversationFile
from lib.tools.astronomy.flat_fielding import flat_field_correct
from lib.tools.astronomy.point_source import detect_sources_csv_bytes
from lib.tools.astronomy.sky_subtraction import subtract_sky_arrays


def sky_subtraction_tool(chat_consumer: "ChatConsumer") -> LLMTool:
    @llm_tool(
        for_whom="assistant",
        required=["object_id", "sky_id"],
        param_descs={
            "object_id": "The file ID of the FITS file containing the object to subtract the sky from",
            "sky_id": "The file ID of the FITS file of the sky to subtract from the object",
        },
    )
    def sky_subtraction(object_id: int, sky_id: int) -> ToolResultDict:
        """
        Subtracts the sky from a FITS image of an object.

        Use this when a user asks you to subtract the sky from an object, and provides the files,
        one of the sky and one of the object.
        Specify the IDs of the files in the parameters.
        """
        from astropy.io import fits as fits_module

        try:
            convo = chat_consumer.db_convo
            object_cf = ConversationFile.objects.filter(id=object_id).first()
            sky = ConversationFile.objects.filter(id=sky_id).first()
            if object_cf is None or sky is None:
                return {"exception": "One or more files do not exist!"}
            if object_cf.conversation != convo or sky.conversation != convo:
                return {"exception": "One or more files do not belong to this conversation!"}
            object_file = object_cf.file
            sky_file = sky.file
            object_data = fits_module.getdata(object_file.open("rb"))
            sky_data = fits_module.getdata(sky_file.open("rb"))
            result = subtract_sky_arrays(object_data, sky_data)
            result_io = io.BytesIO()
            fits_module.writeto(result_io, result, overwrite=True)
            result_io.seek(0)
            result_file = ContentFile(
                result_io.read(), name=f"{object_file.name[:-5]}_sky_subtracted.fits"
            )
            result_conversation_file = ConversationFile(
                file=result_file,
                conversation=convo,
                name=f"{object_file.name[:-5]}_sky_subtracted.fits",
            )
            result_conversation_file.save()
            return {
                "result": "Sky subtracted!",
                "files": [(result_conversation_file.name, result_conversation_file.id)],
            }
        except Exception as e:
            return {"exception": f"An error occurred while subtracting the sky: {str(e)}"}

    return sky_subtraction


def flat_fielding_tool(chat_consumer: "ChatConsumer") -> LLMTool:
    @llm_tool(
        for_whom="assistant",
        required=["science_id", "flat_id"],
        param_descs={
            "science_id": "The file ID of the FITS image to be flat-field corrected",
            "flat_id": "The file ID of the flat-field FITS image to use for correction",
        },
    )
    def flat_fielding(science_id: int, flat_id: int) -> ToolResultDict:
        """
        Applies flat-field correction to a FITS image.

        Use this when a user provides a science image and a flat-field image to correct for
        detector sensitivity variations.
        """
        from astropy.io import fits

        try:
            convo = chat_consumer.db_convo
            science = ConversationFile.objects.filter(id=science_id).first()
            flat = ConversationFile.objects.filter(id=flat_id).first()
            if science is None or flat is None:
                return {"exception": "One or more files do not exist!"}
            if science.conversation != convo or flat.conversation != convo:
                return {"exception": "One or more files do not belong to this conversation!"}
            science_file = science.file
            flat_file = flat.file
            science_data = fits.getdata(science_file.open("rb"))
            flat_data = fits.getdata(flat_file.open("rb"))
            result = flat_field_correct(science_data, flat_data)
            result_io = io.BytesIO()
            fits.writeto(result_io, result, overwrite=True)
            result_io.seek(0)
            result_file = ContentFile(
                result_io.read(), name=f"{science_file.name[:-5]}_flat_corrected.fits"
            )
            result_conversation_file = ConversationFile(
                file=result_file,
                conversation=convo,
                name=f"{science_file.name[:-5]}_flat_corrected.fits",
            )
            result_conversation_file.save()
            return {
                "result": "Flat-fielding applied!",
                "files": [(result_conversation_file.name, result_conversation_file.id)],
            }
        except Exception as e:
            return {"exception": f"An error occurred during flat-fielding: {str(e)}"}

    return flat_fielding


def point_source_detection_tool(chat_consumer: "ChatConsumer") -> LLMTool:
    @llm_tool(
        for_whom="assistant",
        required=["image_id"],
        param_descs={
            "image_id": (
                "The file ID of the sky-subtracted and flat-fielded FITS image to run source detection on."
            )
        },
    )
    def detect_point_sources(image_id: int) -> ToolResultDict:
        """
        Detect point sources in a processed FITS image using DAOStarFinder.
        Apply the method of sigma clipping with sigma=3.0, using DAOStarFinder with fwhm=3.0
        and threshold=5*std

        Use this after sky subtraction and flat-fielding to extract point sources from the image.
        """
        from astropy.io import fits

        try:
            convo = chat_consumer.db_convo
            image = ConversationFile.objects.filter(id=image_id).first()
            if image is None:
                return {"exception": "The file does not exist!"}
            if image.conversation != convo:
                return {"exception": "The file does not belong to this conversation!"}

            image_file = image.file
            data = fits.getdata(image_file.open("rb"))

            count, csv_bytes = detect_sources_csv_bytes(data)
            if count == 0:
                return {"result": "No sources detected."}

            csv_file = ContentFile(
                csv_bytes,
                name=f"{image_file.name[:-5]}_sources.csv",
            )
            result_conversation_file = ConversationFile(
                file=csv_file,
                conversation=convo,
                name=csv_file.name,
            )
            result_conversation_file.save()

            return {
                "result": f"Detected {count} sources.",
                "files": [(result_conversation_file.name, result_conversation_file.id)],
            }
        except Exception as e:
            return {"exception": f"An error occurred during source detection: {str(e)}"}

    return detect_point_sources


__all__ = ["flat_fielding_tool", "point_source_detection_tool", "sky_subtraction_tool"]
