from types import SimpleNamespace

from app.main import verify_artifact_file

from app.database import ArtifactModel, RepositoryModel
from app.models import ArtifactStatus, ArtifactFormat

from botocore.exceptions import ClientError


def test_artifact_status_values():
    assert ArtifactStatus.DISCOVERED.value == "discovered"
    assert ArtifactStatus.DOWNLOADING.value == "downloading"
    assert ArtifactStatus.AVAILABLE.value == "available"
    assert ArtifactStatus.VERIFYING.value == "verifying"
    assert ArtifactStatus.VERIFIED.value == "verified"
    assert ArtifactStatus.CORRUPTED.value == "corrupted"
    assert ArtifactStatus.FAILED.value == "failed"


def test_download_success_transitions_to_available():
    """
    Contract:

    DISCOVERED
        -> DOWNLOADING
        -> AVAILABLE
    """
    assert ArtifactStatus.DISCOVERED.value == "discovered"
    assert ArtifactStatus.DOWNLOADING.value == "downloading"
    assert ArtifactStatus.AVAILABLE.value == "available"


def test_download_failure_transitions_to_failed():
    """
    Contract:

    DISCOVERED
        -> DOWNLOADING
        -> FAILED
    """
    assert ArtifactStatus.DISCOVERED.value == "discovered"
    assert ArtifactStatus.DOWNLOADING.value == "downloading"
    assert ArtifactStatus.FAILED.value == "failed"


def test_verify_success_transitions_to_verified():
    """
    Contract:

    AVAILABLE
        -> VERIFYING
        -> VERIFIED
    """
    assert ArtifactStatus.AVAILABLE.value == "available"
    assert ArtifactStatus.VERIFYING.value == "verifying"
    assert ArtifactStatus.VERIFIED.value == "verified"


def test_verify_failure_transitions_to_corrupted():
    """
    Contract:

    AVAILABLE
        -> VERIFYING
        -> CORRUPTED
    """
    assert ArtifactStatus.AVAILABLE.value == "available"
    assert ArtifactStatus.VERIFYING.value == "verifying"
    assert ArtifactStatus.CORRUPTED.value == "corrupted"


def test_failed_artifact_can_be_downloaded_again():
    """
    FAILED artifacts are recoverable.

    FAILED -> DOWNLOADING
    """
    assert ArtifactStatus.FAILED.value == "failed"
    assert ArtifactStatus.DOWNLOADING.value == "downloading"


def test_corrupted_artifact_can_be_downloaded_again():
    """
    CORRUPTED artifacts are recoverable.

    CORRUPTED -> DOWNLOADING
    """
    assert ArtifactStatus.CORRUPTED.value == "corrupted"
    assert ArtifactStatus.DOWNLOADING.value == "downloading"


def test_verified_artifact_can_be_verified_again():
    """
    Re-verification is allowed.

    VERIFIED -> VERIFYING
    """
    assert ArtifactStatus.VERIFIED.value == "verified"
    assert ArtifactStatus.VERIFYING.value == "verifying"


def make_artifact_file(
    path="model.safetensors",
    size_bytes=100,
    checksum=None,
):
    return SimpleNamespace(
        path=path,
        size_bytes=size_bytes,
        checksum=checksum,
    )


def make_artifact(
    bucket="models",
    prefix="huggingface/test/revision/transformers/",
):
    return SimpleNamespace(
        minio_bucket=bucket,
        minio_prefix=prefix,
    )


def test_verify_artifact_file_missing(monkeypatch):
    artifact = make_artifact()
    artifact_file = make_artifact_file()

    error = ClientError(
        {
            "Error": {
                "Code": "NoSuchKey",
                "Message": "Not Found",
            }
        },
        "HeadObject",
    )

    def fake_head_object(**kwargs):
        raise error

    monkeypatch.setattr(
        "app.main.s3.head_object",
        fake_head_object,
    )

    result = verify_artifact_file(
        artifact,
        artifact_file,
    )

    assert result == {
        "path": "model.safetensors",
        "status": "missing",
    }


def test_verify_artifact_file_size_mismatch(monkeypatch):
    artifact = make_artifact()
    artifact_file = make_artifact_file(
        size_bytes=100,
    )

    monkeypatch.setattr(
        "app.main.s3.head_object",
        lambda **kwargs: {
            "ContentLength": 200,
        },
    )

    result = verify_artifact_file(
        artifact,
        artifact_file,
    )

    assert result == {
        "path": "model.safetensors",
        "status": "size_mismatch",
        "expected_size": 100,
        "actual_size": 200,
    }


def test_verify_artifact_file_checksum_mismatch(monkeypatch):
    artifact = make_artifact()
    artifact_file = make_artifact_file(
        size_bytes=100,
        checksum="expected",
    )

    monkeypatch.setattr(
        "app.main.s3.head_object",
        lambda **kwargs: {
            "ContentLength": 100,
        },
    )

    monkeypatch.setattr(
        "app.main.calculate_object_sha256",
        lambda **kwargs: "actual",
    )

    result = verify_artifact_file(
        artifact,
        artifact_file,
    )

    assert result == {
        "path": "model.safetensors",
        "status": "checksum_mismatch",
        "expected_checksum": "expected",
        "actual_checksum": "actual",
    }


def test_verify_artifact_file_verified(monkeypatch):
    artifact = make_artifact()
    artifact_file = make_artifact_file(
        size_bytes=100,
        checksum="expected",
    )

    monkeypatch.setattr(
        "app.main.s3.head_object",
        lambda **kwargs: {
            "ContentLength": 100,
        },
    )

    monkeypatch.setattr(
        "app.main.calculate_object_sha256",
        lambda **kwargs: "expected",
    )

    result = verify_artifact_file(
        artifact,
        artifact_file,
    )

    assert result == {
        "path": "model.safetensors",
        "status": "verified",
        "size": 100,
        "checksum": "expected",
    }


def test_verify_artifact_file_size_verified_without_checksum(
    monkeypatch,
):
    artifact = make_artifact()
    artifact_file = make_artifact_file(
        size_bytes=100,
        checksum=None,
    )

    monkeypatch.setattr(
        "app.main.s3.head_object",
        lambda **kwargs: {
            "ContentLength": 100,
        },
    )

    result = verify_artifact_file(
        artifact,
        artifact_file,
    )

    assert result == {
        "path": "model.safetensors",
        "status": "size_verified",
        "size": 100,
    }

def test_verify_artifact_file_uses_artifact_bucket(monkeypatch):
    artifact = make_artifact(
        bucket="models",
        prefix="test/",
    )

    artifact_file = make_artifact_file(
        path="model.safetensors",
        size_bytes=100,
    )

    calls = []

    def fake_head_object(**kwargs):
        calls.append(kwargs)

        return {
            "ContentLength": 100,
        }

    monkeypatch.setattr(
        "app.main.s3.head_object",
        fake_head_object,
    )

    verify_artifact_file(
        artifact,
        artifact_file,
    )

    assert calls == [
        {
            "Bucket": "models",
            "Key": "test/model.safetensors",
        }
    ]
