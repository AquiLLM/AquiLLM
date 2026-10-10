"""Validate transient exact source bytes before durable registration."""


def validate_transient_source(payload) -> None:
    from . import source_artifacts as source

    try:
        if payload.source_bytes is None or not payload.source_filename:
            raise source.SourceContractError("source_bytes_and_filename_required")
        source.validate_artifact_ref(payload.source_artifact, data=payload.source_bytes)
        source.validate_archive_lineage(payload.source_artifact, payload.archive_lineage)
        if len(payload.source_containers) != len(payload.archive_lineage):
            raise source.SourceContractError("source_containers_lineage_mismatch")
        for index, step in enumerate(payload.archive_lineage):
            filename, container_bytes, _content_type = payload.source_containers[index]
            if source.sha256(container_bytes).hexdigest() != step.container_sha256:
                raise source.SourceContractError("source_container_bytes_mismatch")
            if index and filename != payload.archive_lineage[index - 1].member_path:
                raise source.SourceContractError("source_container_filename_mismatch")
            child_bytes = (
                payload.source_containers[index + 1][1]
                if index + 1 < len(payload.source_containers)
                else payload.source_bytes
            )
            # ZipInfo ordinals distinguish byte-identical duplicate names.
            with source.zipfile.ZipFile(source.io.BytesIO(container_bytes)) as archive:
                entries = archive.infolist()
                if step.archive_ordinal >= len(entries):
                    raise source.SourceContractError("source_archive_ordinal_missing")
                entry = entries[step.archive_ordinal]
                entry_name = source.canonical_archive_entry_name(
                    entry.orig_filename, entry.filename, is_directory=entry.is_dir()
                )
                if (
                    entry.is_dir()
                    or entry_name != step.member_path
                    or entry.file_size != len(child_bytes)
                ):
                    raise source.SourceContractError("source_archive_member_mismatch")
                with archive.open(entry) as file_obj:
                    actual = file_obj.read(len(child_bytes) + 1)
                if actual != child_bytes:
                    raise source.SourceContractError("source_archive_bytes_mismatch")
        if (
            payload.archive_lineage
            and payload.source_filename != payload.archive_lineage[-1].member_path
        ):
            raise source.SourceContractError("source_member_filename_mismatch")
        if payload.media_artifact is not None:
            if payload.media_bytes is None:
                raise source.SourceContractError("media_bytes_required")
            source.validate_artifact_ref(payload.media_artifact, data=payload.media_bytes)
    except (
        source.SourceContractError,
        source.zipfile.BadZipFile,
        RuntimeError,
        EOFError,
        TypeError,
        ValueError,
    ) as exc:
        raise source.SourceArtifactIntegrityError("source_payload_integrity_failed") from exc
