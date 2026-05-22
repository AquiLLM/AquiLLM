from __future__ import annotations

from types import SimpleNamespace

from apps.chat.services.tool_wiring.documents import _format_related_figure_payloads


class _Collection:
    def user_can_view(self, _user):
        return True


def test_format_related_figure_payloads_exposes_image_urls():
    figure = SimpleNamespace(
        id="fig-1",
        title="Source - Figure 1",
        full_text="OCR text from the figure.",
        extracted_caption="A calibration curve.",
        figure_index=0,
        image_file=SimpleNamespace(name="figure.png"),
        collection=_Collection(),
    )

    payloads = _format_related_figure_payloads([figure], user=object())

    assert payloads == [
        {
            "type": "image",
            "title": "Source - Figure 1",
            "text": "A calibration curve.",
            "image_url": "/aquillm/document_image/fig-1/",
            "figure_index": 1,
        }
    ]
