=============================
Tracking your collections
=============================

Ingestion Monitor
=================

After you submit files, ingestion and chunking run in the background. To see what is still running or finished, open **Ingestion Monitor** under **Utilities** in the left sidebar (you can keep it open while you work on a collection page).

Sub-collections and figures
===========================

.. admonition:: Why this matters

   Content such as **images**, **tables**, and other extracted pieces from a document are often stored as a **sub-collection** nested under that document in **Browse** (for example a row whose type is **collection** and name ends with something like **Figures**). Until those steps complete, the main PDF and its sub-collections may not all be ready for retrieval. Watching the monitor helps you confirm when everything has landed.

Dashboard and links
===================

The ingestion dashboard lists items being processed (figures, pages, and so on).

.. figure:: /_static/images/ingestionMonitor/ingestion_monitor.png
   :alt: Collection Browse table with a PDF and a Figures sub-collection, plus Ingestion Monitor dashboard showing per-figure progress and links
   :width: 800px
   :align: center

   Browse shows the document and its nested sub-collection; the monitor shows live status and links into those ingested pieces.
