from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path


class AttachmentMediaClass(StrEnum):
    IMAGE = "image"
    AUDIO = "audio"
    PDF = "pdf"
    TEXT_DOCUMENT = "text_document"
    BINARY_DOCUMENT = "binary_document"
    VIDEO_NOTE = "video_note"
    VIDEO = "video"


@dataclass(frozen=True, slots=True)
class AttachmentRef:
    attachment_id: str
    media_class: AttachmentMediaClass
    original_name: str | None
    mime_type: str | None
    file_size: int | None
    local_path: Path
    derived_text_path: Path | None = None
    preview_path: Path | None = None
    metadata_json: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "attachment_id": self.attachment_id,
            "media_class": self.media_class.value,
            "original_name": self.original_name,
            "mime_type": self.mime_type,
            "file_size": self.file_size,
            "local_path": str(self.local_path),
            "derived_text_path": str(self.derived_text_path) if self.derived_text_path else None,
            "preview_path": str(self.preview_path) if self.preview_path else None,
            "metadata_json": self.metadata_json,
        }

    @classmethod
    def from_payload(cls, payload: object) -> AttachmentRef | None:
        if not isinstance(payload, dict):
            return None
        attachment_id = payload.get("attachment_id")
        media_class = payload.get("media_class")
        local_path = payload.get("local_path")
        if not isinstance(attachment_id, str) or not attachment_id.strip():
            return None
        if not isinstance(media_class, str) or not media_class.strip():
            return None
        if not isinstance(local_path, str) or not local_path.strip():
            return None
        try:
            normalized_media_class = AttachmentMediaClass(media_class)
        except ValueError:
            return None
        return cls(
            attachment_id=attachment_id,
            media_class=normalized_media_class,
            original_name=payload.get("original_name") if isinstance(payload.get("original_name"), str) else None,
            mime_type=payload.get("mime_type") if isinstance(payload.get("mime_type"), str) else None,
            file_size=payload.get("file_size") if isinstance(payload.get("file_size"), int) else None,
            local_path=Path(local_path),
            derived_text_path=Path(value) if isinstance((value := payload.get("derived_text_path")), str) and value.strip() else None,
            preview_path=Path(value) if isinstance((value := payload.get("preview_path")), str) and value.strip() else None,
            metadata_json=payload.get("metadata_json") if isinstance(payload.get("metadata_json"), str) else None,
        )

    def metadata(self) -> dict[str, object]:
        if not self.metadata_json:
            return {}
        try:
            payload = json.loads(self.metadata_json)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}


@dataclass(frozen=True, slots=True)
class UserTurn:
    text: str | None
    attachments: tuple[AttachmentRef, ...] = ()
    source: str = "telegram"
    source_message_id: int | None = None
    created_at: str | None = None

    def has_content(self) -> bool:
        return bool((self.text and self.text.strip()) or self.attachments)

    def normalized_text(self) -> str | None:
        if self.text is None:
            return None
        normalized = self.text.strip()
        return normalized or None

    def to_payload(self) -> dict[str, object]:
        return {
            "text": self.normalized_text(),
            "attachments": [attachment.to_payload() for attachment in self.attachments],
            "source": self.source,
            "source_message_id": self.source_message_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_payload(cls, payload: object) -> UserTurn | None:
        if not isinstance(payload, dict):
            return None
        text = payload.get("text") if isinstance(payload.get("text"), str) else None
        raw_attachments = payload.get("attachments")
        attachments: list[AttachmentRef] = []
        if isinstance(raw_attachments, list):
            for item in raw_attachments:
                attachment = AttachmentRef.from_payload(item)
                if attachment is not None:
                    attachments.append(attachment)
        turn = cls(
            text=text,
            attachments=tuple(attachments),
            source=payload.get("source") if isinstance(payload.get("source"), str) and payload.get("source").strip() else "telegram",
            source_message_id=payload.get("source_message_id") if isinstance(payload.get("source_message_id"), int) else None,
            created_at=payload.get("created_at") if isinstance(payload.get("created_at"), str) else None,
        )
        return turn if turn.has_content() else None
