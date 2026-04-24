from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import mimetypes
from pathlib import Path
import shutil
from typing import Iterable
from uuid import uuid4

from .input_types import AttachmentMediaClass, AttachmentRef
from .runtime.home import ensure_private_directory, ensure_private_file, set_private_file_permissions


TOPIC_TMP_DIRNAME = "tmp"
TOPIC_TMP_TOPICS_DIRNAME = "topics"
REPO_TMP_DIRNAME = ".turnmux-tmp"
TEXT_EXTRACT_MAX_BYTES = 256 * 1024
TEXT_INLINE_MAX_CHARS = 4_000
TEXT_DOCUMENT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".log",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".swift",
    ".go",
    ".rs",
    ".java",
    ".sh",
    ".zsh",
    ".toml",
    ".ini",
    ".cfg",
    ".sql",
    ".xml",
    ".html",
    ".css",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


@dataclass(frozen=True, slots=True)
class StoredAttachment:
    ref: AttachmentRef
    manifest_path: Path


class AttachmentStore:
    def __init__(self, runtime_home: Path) -> None:
        self.runtime_home = runtime_home.expanduser().resolve(strict=False)
        self.tmp_root = ensure_private_directory(self.runtime_home / TOPIC_TMP_DIRNAME / TOPIC_TMP_TOPICS_DIRNAME)

    def topic_dir(self, chat_id: int, thread_id: int) -> Path:
        return self.tmp_root / str(chat_id) / str(thread_id)

    def store_attachment(
        self,
        chat_id: int,
        thread_id: int,
        *,
        original_name: str | None,
        mime_type: str | None,
        payload: bytes,
        source_message_id: int | None,
        source_kind: str,
        metadata: dict[str, object] | None = None,
    ) -> AttachmentRef:
        topic_dir = ensure_private_directory(self.topic_dir(chat_id, thread_id))
        attachment_id = f"att_{uuid4().hex[:12]}"
        attachment_dir = ensure_private_directory(topic_dir / attachment_id)
        media_class = _classify_attachment(original_name, mime_type, payload, source_kind=source_kind)
        suffix = _safe_suffix(original_name, mime_type, media_class)
        raw_path = attachment_dir / f"raw{suffix}"
        raw_path.write_bytes(payload)
        set_private_file_permissions(raw_path)

        derived_text_path: Path | None = None
        extracted_text: str | None = None
        if media_class == AttachmentMediaClass.TEXT_DOCUMENT:
            extracted_text = _extract_text_document(payload)
            if extracted_text:
                derived_text_path = attachment_dir / "extracted.txt"
                derived_text_path.write_text(extracted_text, encoding="utf-8")
                set_private_file_permissions(derived_text_path)

        manifest_payload = {
            "attachment_id": attachment_id,
            "chat_id": chat_id,
            "thread_id": thread_id,
            "source_message_id": source_message_id,
            "source_kind": source_kind,
            "original_name": original_name,
            "mime_type": mime_type,
            "file_size": len(payload),
            "sha256": sha256(payload).hexdigest(),
            "media_class": media_class.value,
            "local_path": str(raw_path),
            "derived_text_path": str(derived_text_path) if derived_text_path else None,
            "metadata": metadata or {},
        }
        manifest_path = attachment_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        set_private_file_permissions(manifest_path)

        return AttachmentRef(
            attachment_id=attachment_id,
            media_class=media_class,
            original_name=original_name,
            mime_type=mime_type,
            file_size=len(payload),
            local_path=raw_path,
            derived_text_path=derived_text_path,
            metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
        )

    def project_attachment(self, repo_path: Path, *, chat_id: int, thread_id: int, attachment: AttachmentRef) -> AttachmentRef:
        projection_dir = ensure_private_directory(self.topic_projection_dir(repo_path, chat_id, thread_id) / attachment.attachment_id)
        _ensure_repo_tmp_excluded(repo_path)

        projected_raw = projection_dir / attachment.local_path.name
        shutil.copy2(attachment.local_path, projected_raw)
        set_private_file_permissions(projected_raw)

        projected_derived: Path | None = None
        if attachment.derived_text_path and attachment.derived_text_path.exists():
            projected_derived = projection_dir / attachment.derived_text_path.name
            shutil.copy2(attachment.derived_text_path, projected_derived)
            set_private_file_permissions(projected_derived)

        return AttachmentRef(
            attachment_id=attachment.attachment_id,
            media_class=attachment.media_class,
            original_name=attachment.original_name,
            mime_type=attachment.mime_type,
            file_size=attachment.file_size,
            local_path=projected_raw,
            derived_text_path=projected_derived,
            preview_path=attachment.preview_path,
            metadata_json=attachment.metadata_json,
        )

    def clear_topic(self, chat_id: int, thread_id: int, *, repo_path: Path | None = None) -> None:
        topic_dir = self.topic_dir(chat_id, thread_id)
        if topic_dir.exists():
            shutil.rmtree(topic_dir, ignore_errors=True)
        if repo_path is not None:
            projection_dir = self.topic_projection_dir(repo_path, chat_id, thread_id)
            if projection_dir.exists():
                shutil.rmtree(projection_dir, ignore_errors=True)

    def scrub_orphans(self, *, live_topics: Iterable[tuple[int, int]]) -> None:
        keep = {(str(chat_id), str(thread_id)) for chat_id, thread_id in live_topics}
        if not self.tmp_root.exists():
            return
        for chat_dir in self.tmp_root.iterdir():
            if not chat_dir.is_dir():
                continue
            for thread_dir in chat_dir.iterdir():
                if not thread_dir.is_dir():
                    continue
                if (chat_dir.name, thread_dir.name) not in keep:
                    shutil.rmtree(thread_dir, ignore_errors=True)
            if not any(chat_dir.iterdir()):
                shutil.rmtree(chat_dir, ignore_errors=True)

    def topic_projection_dir(self, repo_path: Path, chat_id: int, thread_id: int) -> Path:
        return repo_path / REPO_TMP_DIRNAME / f"turnmux-{chat_id}-{thread_id}"


def inline_excerpt(path: Path | None, *, limit: int = TEXT_INLINE_MAX_CHARS) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    normalized = text.strip()
    if not normalized:
        return None
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "..."


def _ensure_repo_tmp_excluded(repo_path: Path) -> None:
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        return
    exclude_path = git_dir / "info" / "exclude"
    ensure_private_file(exclude_path)
    marker = f"/{REPO_TMP_DIRNAME}/"
    try:
        current = exclude_path.read_text(encoding="utf-8")
    except OSError:
        current = ""
    if marker in current.splitlines():
        return
    new_content = current
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"
    new_content += marker + "\n"
    exclude_path.write_text(new_content, encoding="utf-8")
    set_private_file_permissions(exclude_path)


def _classify_attachment(
    original_name: str | None,
    mime_type: str | None,
    payload: bytes,
    *,
    source_kind: str,
) -> AttachmentMediaClass:
    if source_kind == "photo":
        return AttachmentMediaClass.IMAGE
    if source_kind == "video_note":
        return AttachmentMediaClass.VIDEO_NOTE
    if source_kind == "video":
        return AttachmentMediaClass.VIDEO
    if source_kind in {"voice", "audio"}:
        return AttachmentMediaClass.AUDIO

    suffix = Path(original_name).suffix.lower() if original_name else ""
    normalized_mime = (mime_type or "").lower()
    if normalized_mime.startswith("image/") or suffix in IMAGE_EXTENSIONS:
        return AttachmentMediaClass.IMAGE
    if normalized_mime == "application/pdf" or suffix == ".pdf":
        return AttachmentMediaClass.PDF
    if normalized_mime.startswith("text/") or suffix in TEXT_DOCUMENT_EXTENSIONS or _looks_like_text(payload):
        return AttachmentMediaClass.TEXT_DOCUMENT
    return AttachmentMediaClass.BINARY_DOCUMENT


def _safe_suffix(original_name: str | None, mime_type: str | None, media_class: AttachmentMediaClass) -> str:
    suffix = Path(original_name).suffix.lower() if original_name else ""
    if suffix:
        return suffix
    guessed = mimetypes.guess_extension(mime_type or "")
    if guessed:
        return guessed
    defaults = {
        AttachmentMediaClass.IMAGE: ".bin",
        AttachmentMediaClass.AUDIO: ".bin",
        AttachmentMediaClass.PDF: ".pdf",
        AttachmentMediaClass.TEXT_DOCUMENT: ".txt",
        AttachmentMediaClass.BINARY_DOCUMENT: ".bin",
        AttachmentMediaClass.VIDEO_NOTE: ".mp4",
        AttachmentMediaClass.VIDEO: ".mp4",
    }
    return defaults[media_class]


def _looks_like_text(payload: bytes) -> bool:
    sample = payload[: min(len(payload), 4096)]
    if not sample:
        return False
    if b"\x00" in sample:
        return False
    printable = 0
    for value in sample:
        if value in (9, 10, 13) or 32 <= value <= 126:
            printable += 1
    return printable / len(sample) >= 0.85


def _extract_text_document(payload: bytes) -> str | None:
    capped = payload[:TEXT_EXTRACT_MAX_BYTES]
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            decoded = capped.decode(encoding)
        except UnicodeDecodeError:
            continue
        normalized = decoded.strip()
        if normalized:
            return normalized
    if _looks_like_text(capped):
        normalized = capped.decode("utf-8", errors="replace").strip()
        return normalized or None
    return None
