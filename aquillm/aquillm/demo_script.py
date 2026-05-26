"""Canned conversation served by FakeInterface when LLM_CHOICE=FAKE.

The list below is consumed one entry per get_message() call, in order.
Plain strings are sent as end_turn text responses; tool_call_response(...)
entries emit a real tool call that the chat consumer executes against the
configured tool wiring. The provider clamps to the final entry once the
script is exhausted, so the demo never crashes if the user keeps chatting.

Story arc (GX 349+2 paper ingested in a selected collection the whole
time, assistant calls vector_search on every turn):

  The paper redefines "CCD" to mean *color-color diagram* on page 1
  ("trace the Z shape on the color-color diagram (CCD)..."), but most
  models default to CCD = charge-coupled device, the X-ray imaging
  detector. The user is asking a question that only makes sense under
  the paper's redefinition.

  Turn 1 — vector_search returns chunks that use CCD in the paper's
           color-color-diagram sense, but the assistant pattern-matches
           it to charge-coupled device and answers confidently-wrong,
           treating "CCD" as the detector and saying classification
           doesn't come from detector hardware.

  (off-screen, the user opens the collection and adds a Collection Note
   disambiguating the two meanings of CCD. That note gets injected into
   the next vector_search tool result.)

  Turn 2 — vector_search runs again; the note steers the assistant to
           the right meaning, and it gives the Z-source classification
           answer the paper actually contains.

Suggested Collection Note to paste into the collection between turns:

    Acronym disambiguation: in this paper "CCD" is shorthand for
    color-color diagram — the classic Z-source classification tool
    that plots hardness ratios against each other to reveal the
    source's Z-shaped track. It is NOT charge-coupled device, the
    detector. When the user asks about "the CCD" anywhere in this
    paper, interpret it as color-color diagram and look at the
    Z-source classification material in section 1 and the
    hardness-intensity discussion in section 3.1 / Figure 1.

Recommended question to type both times:

    "How do the authors use the CCD to classify GX 349+2?"
"""
from .llm import tool_call_response


DEMO_RESPONSES = [
    # ----- Turn 1: vector_search runs. The chunks contain CCD in the
    #               color-color-diagram sense, but no Collection Note
    #               warns the assistant about that overload.
    tool_call_response(
        "vector_search",
        {"search_string": "CCD classify GX 349+2 Z-source branch", "top_k": 6},
        text="Let me check the collection.",
    ),

    # ----- Turn 1 post-tool: confidently-wrong answer, treating CCD
    #       as the imaging detector (the model's default reading).
    (
        "The CCD itself doesn't really classify sources — a "
        "charge-coupled device is just the imaging detector on the "
        "focal plane (here, the one in Swift/XRT; IXPE and NuSTAR use "
        "different detector technologies). Source classification — "
        "Z-type vs atoll-type, Sco-like vs Cyg-like, which branch the "
        "source is on — is a *data-analysis* step, not a property of "
        "the detector. You take the photon stream off the CCD, build "
        "spectra and light curves, compute hardness ratios, and *then* "
        "classify.\n\n"
        "Several retrieved chunks describe GX 349+2 as an \"Sco-like\" "
        "Z-source that traces branches called FB and NB, and Figure 1 "
        "is described as plotting hardness against intensity. But none "
        "of that is the CCD doing the classifying — it's the authors' "
        "analysis of the data the CCD recorded. If you meant something "
        "more specific by \"CCD\" — a particular plot, maybe — let me "
        "know and I'll re-search."
    ),

    # ----- Turn 2: vector_search runs again. The Collection Note
    #              added off-screen tells the assistant that CCD in
    #              this paper means color-color diagram.
    tool_call_response(
        "vector_search",
        {
            "search_string": "color-color diagram Z-source flaring branch normal branch GX 349+2 Sco-like",
            "top_k": 6,
        },
        text="Got it — re-running with the collection note in mind; the paper's CCD is the color-color diagram, not the detector.",
    ),

    # ----- Turn 2 post-tool: correct, paper-specific answer.
    (
        "Apologies for the earlier confusion — the paper uses **CCD as "
        "shorthand for *color-color diagram***, not charge-coupled "
        "device. With that disambiguation, the question has a real "
        "answer.\n\n"
        "The color-color diagram is the standard classification tool "
        "for low-mass X-ray binaries: Z-type sources trace a "
        "Z-shaped track across three branches — the **horizontal branch "
        "(HB)**, the **normal branch (NB)**, and the **flaring branch "
        "(FB)** — joined by a *hard apex* (HB↔NB) and a *soft apex* "
        "(NB↔FB). Atoll-type sources, by contrast, trace a C-shape on "
        "much longer timescales.\n\n"
        "Within the Z-class, GX 349+2 is identified as **Sco-like** "
        "rather than Cyg-like because it shows strong NB and FB but "
        "essentially no horizontal branch (Hasinger & van der Klis "
        "1989; Kuulkers & van der Klis 1998). The authors use the "
        "related hardness-intensity diagram (HID) in Figure 1 — "
        "hardness ratio (4-8 keV / 2-4 keV) on the y-axis, 2-8 keV "
        "intensity on the x-axis — to identify the FB (red points, "
        "harder and brighter) and NB (blue points, softer and "
        "dimmer), and to track GX 349+2 transitioning FB → NB → FB "
        "during the IXPE observation.\n\n"
        "So: the *CCD* (color-color diagram) is what classifies "
        "GX 349+2 as a Z-source, and the absence of the horizontal "
        "branch on that diagram is what further classifies it as "
        "Sco-like."
    ),
]


__all__ = ["DEMO_RESPONSES"]
